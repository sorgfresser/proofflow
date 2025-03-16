import gc
from collections.abc import Callable
from dataclasses import dataclass
from math import exp, log
import time
from collections import defaultdict
from typing import Tuple, List, Union, Dict, Any, Iterator, Optional
from torch import nn
import torch
from torch_scatter import scatter
from lean_repl_py import LeanREPLHandler, LeanREPLNextProofState, LeanREPLProofState, LeanREPLAsyncHandler
from proofflow.model.ffm import FFM
from proofflow.data import parse_json, LEAN_DOJO_PATH, TheoremDataset, TrainSampleDataset, TrainingSample, Theorem, \
    UnknownMetaVariableError, ProofStateDataset
from pathlib import Path
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader
from proofflow.policy import Policy, MambaLMHeadModelWrapper, MambaPolicy
from transformers import PreTrainedTokenizerFast
import torch.optim as optim
from math import sqrt
import wandb
from mamba_ssm.models.config_mamba import MambaConfig
from tqdm import tqdm
from argparse import ArgumentParser
import asyncio

MAX_TRAJ_LEN = 10
MAX_OUTPUT_LEN = 20
GET_STATE_EVERY = 2  # we want semantically similar proofs to have the same states. Increasing this helps to do that
TEMPERATURE = 1


class ModelBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, memory_size: int, content_size: int,
                 dropout: float):
        super().__init__()
        self.act = nn.LeakyReLU()
        self.norm1 = nn.LayerNorm(input_size)
        self.ffm = FFM(input_size, hidden_size, memory_size=memory_size, context_size=content_size,
                       output_size=input_size)
        self.linear = nn.Linear(input_size, output_size)
        self.norm2 = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x, _ = self.ffm(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.norm2(x)
        x = self.act(x)
        return x


class Model(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.act = nn.LeakyReLU()
        self.emb = nn.Embedding(vocab_size, 20)
        self.norm = nn.LayerNorm(20)
        self.dropout = nn.Dropout(p=0.1)
        # self.block1 = ModelBlock(20, 20, 64, 3, 3, 0.1)
        # self.block2 = ModelBlock(20, 20, 64, 3, 3, 0.1)
        self.block3 = ModelBlock(20, vocab_size, 64, 3, 3, 0.1)
        self.z_head = (nn.Linear(20, 1, bias=True),)  # hack to not register as parameter
        self.back_head = nn.Linear(20, vocab_size, bias=False)

    def backbone(self, x):
        if x.shape == 1:
            x = x[None]
        assert len(x.shape) == 2
        x = self.emb(x)  # [batch_dim, seq_len, emb_dim]
        x = self.norm(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.backbone(x)
        return self.block3(x)

    def p_b(self, input_ids):
        hidden_states = self.backbone(input_ids)
        back_logits = self.back_head(hidden_states)
        return back_logits

    def log_z(self, input_ids, attention_mask):
        hidden_states = self.backbone(input_ids)
        # Get the last one that has attention one
        last_indices = attention_mask.sum(1) - 1
        hidden_states = hidden_states[torch.arange(hidden_states.shape[0]), last_indices]
        lm_logits = self.z_head[0](hidden_states)
        return lm_logits

    def get_non_z_params(self):
        return [i for i in self.parameters() if all(id(i) != id(j) for j in self.get_z_params())]

    def get_z_params(self):
        return self.z_head[0].parameters()


def collate_train_samples(batch: list[TrainingSample]):
    return batch


def evaluate(policy: Policy, data: TrainSampleDataset, eval_batch_size: int = 64) -> dict[str, float]:
    metrics = defaultdict(float)
    print("Evaluating")
    with torch.no_grad():
        for batch in tqdm(DataLoader(data, batch_size=eval_batch_size, shuffle=True, collate_fn=collate_train_samples)):
            metrics_batch = policy.evaluate_batch(batch)
            for key, value in metrics_batch.items():
                metrics[key] += value
        for key in metrics:
            metrics[key] /= len(data) / eval_batch_size
    return metrics


def train_loop(policy: Policy, data: TrainSampleDataset, optimizer: optim.Optimizer, gradient_accumulation_steps: int,
               batch_size: int, eval_steps: int, valid_data: TrainSampleDataset, checkpoint_path: Path,
               eval_batch_size: int = 4, epochs: int = 3):
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_train_samples)
    policy.model.train()
    optimizer.zero_grad()
    current_step = 0
    print(f"Saving model to {checkpoint_path}")
    policy.save(checkpoint_path)
    for epoch in range(epochs):
        for batch in tqdm(data_loader):
            loss = policy.train_batch(batch) / gradient_accumulation_steps
            wandb.log({"train/loss": loss, "epoch": current_step / len(data_loader)}, step=current_step)
            loss.backward()
            if (current_step + 1) % gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if (current_step + 1) % eval_steps == 0:
                metrics = evaluate(policy, valid_data, eval_batch_size)
                metrics = {f"validation/{key}": value for key, value in metrics.items()}
                wandb.log(metrics, step=current_step)
                print(metrics)
            current_step += 1
        print(f"Epoch {epoch} done")
        print(f"Saving model to {checkpoint_path}")
        policy.save(checkpoint_path)
    print("Training done")
    metrics = evaluate(policy, valid_data, eval_batch_size)
    metrics = {f"validation/{key}": value for key, value in metrics.items()}
    wandb.log(metrics)


def huber_loss(x, beta=1, i_delta=4):
    ax = torch.abs(x)
    return torch.where(ax <= beta, 0.5 * x * x, beta * (ax - beta / 2)) * i_delta


# retrieve the initial theorems
def get_start_theorems(path):
    for thm in tqdm(parse_json(path)):
        # Dataset errors, i.e. no tactics were traced
        if not thm.traced_tactics:
            continue
        # Another dataset error, they also globbed the lake files, i.e. mathlib's dependencies
        if ".lake" in thm.file_path:
            continue
        yield thm


@dataclass
class Node:
    proof_state: str
    parent: Union["Node", None] = None
    parent_tactic: str = ""
    children_for_tactics: Dict[str, "Node"] = None
    visit_counts: Dict[str, int] = None
    total_action_values: Dict[str, float] = None
    times: Dict[str, float] = None
    has_been_expanded: bool = False
    branch_is_done: bool = False
    previous_states: List[str] = None
    proof_state_idx: int = -1
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.children_for_tactics = {} if self.children_for_tactics is None else self.children_for_tactics
        self.visit_counts = {} if self.visit_counts is None else self.visit_counts
        self.total_action_values = {} if self.total_action_values is None else self.total_action_values
        self.times = {} if self.times is None else self.times
        self.previous_states = [] if self.previous_states is None else self.previous_states
        self.metadata = {} if self.metadata is None else self.metadata

    def expand(self, tactics: List[str], proof_states: List[str], times: List[float], values: List[float],
               indices: List[int]):
        assert not self.has_been_expanded
        assert not self.children_for_tactics
        self.has_been_expanded = True
        if not tactics:
            self.backup_branch_done()
            return

        for idx, tactic in enumerate(tactics):
            # Duplicates in tactics are possible
            if tactic not in self.children_for_tactics:
                self.children_for_tactics[tactic] = Node(proof_states[idx], parent=self, parent_tactic=tactic,
                                                         previous_states=self.previous_states + [self.proof_state],
                                                         proof_state_idx=indices[idx], metadata=self.metadata)
                self.times[tactic] = times[idx]
                self.total_action_values[tactic] = values[idx]
                self.visit_counts[tactic] = 1
        self.backup()

    def backup_branch_done(self):
        self.branch_is_done = True
        if self.parent is None:
            return
        if all(child.branch_is_done for child in self.parent.children_for_tactics.values()):
            self.parent.backup_branch_done()

    def action_values(self):
        return {tactic: self.total_action_values[tactic] / self.visit_counts[tactic] for tactic in self.visit_counts}

    def node_value(self):
        return max(self.action_values().values())  # TODO: can try various operations here

    def backup(self):
        if self.parent is None:
            return
        self.parent.visit_counts[self.parent_tactic] += 1
        self.parent.total_action_values[self.parent_tactic] += self.node_value()
        self.parent.backup()

    def best_action(self):
        return max(self.action_values(), key=self.action_values().get)

    def visit_count(self):
        return sum(self.visit_counts.values())

    def best_action_policy(self):
        # Computes UCB for now, filter ones where branch is not done
        valid_tactics = [tactic for tactic in self.children_for_tactics if
                         not self.children_for_tactics[tactic].branch_is_done]
        return max(self.action_values(), key=lambda x: self.action_values()[x] + 1.0 * sqrt(
            log(self.visit_count()) / self.visit_counts[x]) if x in valid_tactics else -float("inf"))

    def select(self):
        if not self.has_been_expanded:
            return self
        action = self.best_action_policy()
        if self.children_for_tactics[action].branch_is_done:
            assert self.branch_is_done  # This has to imply -inf was chosen, so we should be done
            return self
        return self.children_for_tactics[self.best_action_policy()].select()


class MCTS:
    def __init__(self, root: Node):
        self.root = root
        self.proof = ""
        self.time = 0.0
        self.step_count = 0
        self.done = False
        self.last_tactic = ""
        self.solved = False

    def move(self):
        if self.done:
            return
        best_tactic = self.root.select().best_action()
        self.proof += best_tactic + "\n"
        self.time += self.root.times[best_tactic]
        self.root = self.root.children_for_tactics[best_tactic]
        self.root.parent = None  # speedup memory release
        self.step_count += 1
        self.last_tactic = best_tactic
        gc.collect()

    def select(self) -> Node:
        node = self.root
        while node != node.select():
            node = node.select()
        if node.branch_is_done:
            self.done = True
        return node


async def _get_start_states(start_loader: Iterator, handler_factory: Callable[[], LeanREPLAsyncHandler]) -> Tuple[
    List[LeanREPLProofState], List[MCTS], List[LeanREPLAsyncHandler]]:
    batch = next(start_loader)
    thms, paths = [elem[0] for elem in batch], [elem[1] for elem in batch]
    handlers = [handler_factory() for _ in paths]
    # Start all REPLs
    proof_state_futures = [handler.unpickle_proof_state(path) for handler, path in zip(handlers, paths)]
    # Gather
    proof_states = await asyncio.gather(*proof_state_futures)
    proof_states = [proof_state for proof_state, env in proof_states]
    # Unlink them all, not needed anymore
    for path in paths:
        path.unlink()
    assert all(len(proof_state.goals) == 1 for proof_state in proof_states)
    nodes = [Node(proof_state.goals[0], proof_state_idx=proof_state.proof_state,
                  metadata={"theoremname": thm.full_name, "theoremfile": thm.file_path}) for proof_state, thm in
             zip(proof_states, thms)]
    return proof_states, [MCTS(node) for node in nodes], handlers


def _env_expand(handler: LeanREPLAsyncHandler, tactics: List[str], proof_state_indices: List[int]):
    proven = []
    invalid = []
    indices = []
    goals = []
    times = []
    for tactic, proof_state_idx in zip(tactics, proof_state_indices, strict=True):
        curr_time = time.perf_counter()
        asyncio.run(handler.send_tactic(tactic, proof_state_idx))
        response, _ = asyncio.run(handler.receive_json())
        times.append(time.perf_counter() - curr_time)
        if "message" in response and response["message"].startswith("Lean error"):
            invalid.append(True)
            proven.append(False)
            indices.append(None)
            goals.append(None)
            continue
        assert isinstance(response, LeanREPLNextProofState)
        proven.append(not response.goals)
        invalid.append(False)
        if not response.goals:
            goals.append(None)
        else:
            goals.append(response.goals[0])  # only need one goal, it has all information
        indices.append(response.proof_state)
    return proven, invalid, indices, goals, times


async def _process_single_env(handler: LeanREPLAsyncHandler, tactics: List[str], proof_state_indices: List[int],
                              proven: List[bool], invalid: List[bool], indices: List[Optional[int]],
                              goals: List[Optional[List[str]]], times: List[float]):
    assert len(proven) == len(invalid) == len(indices) == len(goals) == 0
    for tactic, proof_state_idx in zip(tactics, proof_state_indices, strict=True):
        curr_time = time.perf_counter()
        await handler.send_tactic(tactic, proof_state_idx)
        response, _ = await handler.receive_json()
        times.append(time.perf_counter() - curr_time)
        if "message" in response and response["message"].startswith("Lean error"):
            invalid.append(True)
            proven.append(False)
            indices.append(None)
            goals.append(None)
            continue
        assert isinstance(response, LeanREPLNextProofState)
        proven.append(not response.goals)
        invalid.append(False)
        if not response.goals:
            goals.append(None)
        else:
            goals.append(response.goals[0])  # only need one goal, it has all information
        indices.append(response.proof_state)


async def _process_all_envs(handlers: List[LeanREPLAsyncHandler], tactics: List[List[str]],
                            proof_state_indices: List[List[int]]):
    proven = [[] for _ in handlers]
    invalid = [[] for _ in handlers]
    indices = [[] for _ in handlers]
    goals = [[] for _ in handlers]
    times = [[] for _ in handlers]
    tasks = []
    for i in range(len(handlers)):
        tasks.append(
            _process_single_env(handlers[i], tactics[i], proof_state_indices[i], proven[i], invalid[i], indices[i],
                                goals[i], times[i]))
    await asyncio.gather(*tasks)
    return proven, invalid, indices, goals, times


def _envs_expand(
        handlers: list[LeanREPLAsyncHandler],
        tactics: list[list[str]],
        proof_state_indices: list[list[int]]
):
    """Sends one tactic at a time per handler"""
    assert len(handlers) == len(tactics) == len(proof_state_indices)
    loop = asyncio.get_event_loop()
    assert not loop.is_closed()
    proven, invalid, indices, goals, times = loop.run_until_complete(_process_all_envs(handlers, tactics, proof_state_indices))
    return proven, invalid, indices, goals, times


def _compute_rewards(proven: List[bool], invalid: List[bool], times: List[float]):
    rewards = []
    for p, i, t in zip(proven, invalid, times):
        if i:
            rewards.append(log(0.01))
        elif p:
            rewards.append(log(10) - t / 5)  # proof complete
        else:
            rewards.append(log(0.1) - t / 5)  # ongoing = small reward
    return rewards


class BackgroundDataLoader:
    """
    Wraps an existing PyTorch DataLoader in a background thread
    that prefetches batches and places them into a queue.
    """

    def __init__(self, dataloader: DataLoader, handler_factory: Callable[[], LeanREPLAsyncHandler],
                 max_prefetch: int = 2):
        self.dataloader_iter = iter(dataloader)
        self.queue: asyncio.Queue[Tuple[List[MCTS], List[LeanREPLAsyncHandler]]] = asyncio.Queue(maxsize=max_prefetch)
        self.stop_flag = False
        self.handler_factory = handler_factory

    async def _run_async(self):
        """The async producer coroutine that fetches data and puts it into the queue."""
        while not self.stop_flag:
            try:
                _, start_states, handlers = await _get_start_states(self.dataloader_iter, self.handler_factory)
            except StopIteration:
                raise RuntimeError("Should not happen")
            await self.queue.put((start_states, handlers))

    async def start_background(self):
        """Start the background task that fetches data."""
        self.task = asyncio.create_task(self._run_async())

    async def get_next_batch(self):
        """Async method to get a batch (waits if queue is empty)."""
        return await self.queue.get()

    async def stop(self):
        """Stop the loader."""
        self.stop_flag = True
        # Empty queue to avoid deadlock (because of putting)
        while not self.queue.empty():
            await self.queue.get()
        await self.task


def train_gflownet(
        policy: Policy,
        start_loader: DataLoader,
        precomputed_trajectories: List[Tuple[List[List[int]], List[List[int]]]],
        handler_factory: Callable[[], LeanREPLAsyncHandler],
        # these are the human-written trajectories
        repo_path: Path,
        optimizer: optim.Optimizer,
        z_optimizer: optim.Optimizer,
        gradient_accumulation_steps: int,
        batch_size_replay: int,
        batch_size_sampled: int,
        rounds: int,
        device: str,
        replay_buffer_len: int = 1_000,
        max_retries: int = 5,
        search_time: int = 100
):
    assert precomputed_trajectories
    policy.model.train()

    optimizer.zero_grad()
    z_optimizer.zero_grad()

    replay_buffer = [None] * replay_buffer_len
    replay_end, replay_saturated = 0, False

    tb_loss_agg = back_loss_agg = 0

    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bg_loader = BackgroundDataLoader(start_loader, handler_factory)
    loop.run_until_complete(bg_loader.start_background())

    for r in tqdm(range(rounds)):
        # Reset the handler to avoid memory leaks
        with torch.no_grad():
            # 0. add new trajectories to the replay buffer
            start_time = time.perf_counter()
            start_states, handlers = loop.run_until_complete(bg_loader.get_next_batch())
            print(f"Get start states time: {time.perf_counter() - start_time}")
            # If the whole batch was invalid, quite unlikely, but possible
            if not start_states:
                continue
            action_trajectories = [[] for __ in start_states]  # list of actions for each proof
            state_trajectories = [[] for __ in start_states]  # list of GFlowNet states for each proof
            proof_state_history = [[node.root.proof_state] for node in
                                   start_states]  # list of proof states for each proof
            log_rewards = [log(0.1)] * len(start_states)
            times = [0] * len(start_states)
            done = [False] * len(start_states)
            idx = 0
            while not all([node.done for node in start_states]) and idx < MAX_TRAJ_LEN:
                try:
                    start_time = time.perf_counter()
                    next_tactic_time = 0
                    for _ in range(search_time):
                        if all(node.done for node in start_states):
                            print(f"Breaking after {_} steps!")
                            break
                        currents = []
                        for node in start_states:
                            if node.done:
                                continue
                            current = node.select()
                            assert not current.has_been_expanded
                            assert not current.branch_is_done
                            currents.append(current)
                        end_states = [current.proof_state for current in currents]
                        histories = [current.previous_states for current in currents]
                        tactic_start_time = time.perf_counter()
                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            tactic_strings, _, _ = policy.next_tactics_int(end_states, max_retries, None, histories,temperature=1)
                        next_tactic_time += time.perf_counter() - tactic_start_time
                        current_handlers = [handlers[node_idx] for node_idx in range(len(start_states)) if not start_states[node_idx].done]
                        proven, invalid, indices, goals, times_current = _envs_expand(current_handlers, tactic_strings,
                                                  [[current.proof_state_idx] * len(tactic_strings[current_idx]) for current_idx, current in enumerate(currents)],)
                        current_idx = 0
                        print("Tactic strings", tactic_strings)
                        print("Proven, invalid etc", proven, invalid, indices, goals, times_current)
                        for i, node in enumerate(start_states):
                            if node.done:
                                continue
                            rewards = _compute_rewards(proven[current_idx], invalid[current_idx], times_current[current_idx])
                            # Only passes on valid tactics to expand, we might want to change this
                            tactics = [t for t, p in zip(tactic_strings[current_idx], invalid) if not p]
                            goals_node = [g for g, p in zip(goals[current_idx], invalid) if not p]
                            times_current_node = [t for t, p in zip(times_current[current_idx], invalid) if not p]
                            indices_node = [index for index, p in zip(indices[current_idx], invalid) if not p]
                            rewards = [r for r, p in zip(rewards, invalid) if not p]
                            currents[current_idx].expand(tactics, goals_node, times_current_node, rewards, indices_node)
                            if any(proven[current_idx]):
                                node.done = True
                                node.solved = True
                                node.last_tactic = tactic_strings[current_idx][proven[current_idx].index(True)]
                            # Edge case, if we only have invalid tactics, there is no way to continue
                            elif node.root.branch_is_done:
                                node.done = True
                                node.solved = False
                            current_idx += 1
                except Exception as e:
                    print(f"Error in MCTS: {e}")
                    print(f"Node: {node.root.proof_state}")
                    print(f"Proof: {node.proof}")
                    print(f"Current: {current.proof_state}")
                    print(f"Current previous states: {current.previous_states}")
                    print(f"Current tactic strings: {tactic_strings}")
                    print(f"Start states: {[state.root.proof_state for state in start_states]}")
                    print(f"Current metadata: {current.metadata}")
                    print(f"Node metadata: {node.root.metadata}")
                    strings = []
                    node = current
                    while node.parent is not None:
                        strings.append(node.parent_tactic)
                        node = node.parent
                    current_proof = "\n".join(reversed(strings))
                    print(f"Current proof: {current_proof}")
                    raise e
                print(f"MCTS time: {time.perf_counter() - start_time}")
                print(f"Next tactic time: {next_tactic_time}")
                # Actual MCTS move after trying a few nodes
                for node in start_states:
                    node.move()
                # Fill trajectories with the latest move, update rewards
                prompts = [policy._build_prompt(node.root.proof_state, None, node.root.previous_states) for node in
                           start_states]
                for i, node in enumerate(start_states):
                    if done[i]: continue
                    state_trajectories[i].append(prompts[i])
                    # Observation: we do not update last tactic in case of an invalid MCTS, so this will simply repeat the tactic before the invalid proof state
                    action_trajectories[i].append(
                        policy.tokenizer.encode(node.last_tactic) + [policy.tokenizer.eos_token_id])
                    proof_state_history[i].append(node.root.proof_state)
                    if node.done and node.solved:
                        done[i] = True
                        state_trajectories[i].append([policy.successful_proof_token])
                        log_rewards[i] = log(10) - times[i] / 5
                    elif node.done:
                        state_trajectories[i].append([policy.invalid_proof_token])
                        log_rewards[i] = log(0.01)
                        done[i] = True
                # Update replay buffer
                for i, t in enumerate(zip(state_trajectories, action_trajectories, log_rewards)):
                    # Need to loose references here, otherwise we append incomplete tokens also for the trajectories
                    state, action, reward = t[0].copy(), t[1].copy(), t[2]
                    if not start_states[i].done:
                        state.append([policy.incomplete_proof_token])
                    replay_buffer[replay_end] = (state, action, reward)
                    replay_end += 1
                    if replay_end >= replay_buffer_len:
                        replay_saturated = True
                        replay_end = 0

                # compute next states and rewards
                # actions: List[Union[List[int], None]] = [None] * batch_size_replay
                # j = 0
                # for i in range(batch_size_replay):
                #     if done[i]:
                #         continue
                #
                #     state_trajectories[i].append(prompts[i])
                #
                #     tactic_times = [0] * len(tactic_strings[j])
                #     tactic_id = 0
                #     for tactic_id, (tactic_string, action) in enumerate(zip(tactic_strings[j], tactic_codes[j])):
                #         tactic_times[tactic_id] = time.perf_counter()
                #         handler.send_tactic(tactic_string, envs[i].proof_state)
                #         response, _ = handler.receive_json()
                #         tactic_times[tactic_id] = time.perf_counter() - tactic_times[tactic_id]
                #
                #         has_error = "message" in response and response["message"].startswith("Lean error")
                #         has_error = has_error or "messages" in response and any(
                #             m.severity == "error" for m in response["messages"])
                #
                #         if not has_error:  # TODO: here we just take the first one without an error. We should try something smarter like tree search
                #             break
                #     actions[j] = action
                #     times[i] += tactic_times[tactic_id]
                # if has_error:
                #     log_rewards[i] = log(0.01)  # proof failed
                #     done[i] = True
                #     state_trajectories[i].append([policy.invalid_proof_token])
                #     j += 1
                #     continue
                # assert isinstance(response, LeanREPLNextProofState)
                # assert actions[j] is not None
                # goals = response.goals
                # action_trajectories[i] = actions[j]
                # proof_state_history[i] = "\n".join(goals)

                # if not goals:
                #     proof_length = len(
                #         actions[j])  # TODO: test some different reward formulations (e.g. computation time)
                #     log_rewards[i] = log(10) + log(1 - exp(-proof_length / 5))  # proof complete
                #     log_rewards[i] = log(10) + log(1 - exp(-times[i] / 5))
                #     done[i] = True
                #     state_trajectories[i].append([policy.successful_proof_token])
                # j += 1

                idx += 1

                # for t in zip(state_trajectories, action_trajectories):
                #     if not done[i]:
                #         t[0].append([policy.incomplete_proof_token])
                #
                #     replay_buffer[replay_end] = t
                #     replay_end += 1
                #
                #     if replay_end >= replay_buffer_len:
                #         replay_saturated = True
                #         replay_end = 0

            # 1. randomly sample from the replay buffer and from human trajectories
            end_idx = replay_buffer_len if replay_saturated else replay_end
            idxs_replay = torch.randint(0, end_idx, (batch_size_replay,))
            idxs_precomputed = torch.randint(0, len(precomputed_trajectories), (batch_size_sampled,))
            trajs = [replay_buffer[i] for i in idxs_replay] + [precomputed_trajectories[i] for i in idxs_precomputed]

        # TODO: check this isn't always 0
        # print(loss.grad) if loss is not None else None

        # 2. call the model on each trajectory

        starting_states = [states[0] for states, _, _ in trajs]
        z_prompts = starting_states
        z_padded_prompts = policy.tokenizer.pad({"input_ids": z_prompts}, padding_side="right", return_tensors="pt")
        log_z_inputs = z_padded_prompts.input_ids.to(device)
        log_z_attention_mask = z_padded_prompts.attention_mask.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            log_z = policy.model.log_z(log_z_inputs, log_z_attention_mask)  # some duplicate computation happening here

        traj_lens = torch.tensor([len(actions) for _, actions, _ in trajs], device=device)

        # stack each trajectory so we can use torch_scatter
        log_p_f = torch.zeros(traj_lens.sum(), device=device)
        log_p_b = torch.zeros(traj_lens.sum(), device=device)

        idx = 0
        # for each action, compute the sum of token log probs with prev state (p_f) and next state (p_b)
        prev_states = [state for states, _, _ in trajs for state in states[:-1]]
        next_states = [state for states, _, _ in trajs for state in states[1:]]
        actions = [action for _, actions, _ in trajs for action in actions]
        fwd_inputs = [state + action for state, action in zip(prev_states, actions, strict=True)]
        bck_inputs = [state + action for state, action in zip(next_states, actions)]

        fwd_padded = policy.tokenizer.pad({"input_ids": fwd_inputs}, padding_side="right", return_tensors="pt")
        bck_padded = policy.tokenizer.pad({"input_ids": bck_inputs}, padding_side="right", return_tensors="pt")

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            fwd_outputs = policy.model(fwd_padded.input_ids.to(device))
            bck_outputs = policy.model.p_b(bck_padded.input_ids.to(device))
            # Fill log probs for tactic tokens
            fwd_log = torch.log_softmax(fwd_outputs, dim=-1)
            bck_log = torch.log_softmax(bck_outputs, dim=-1)
            for idx in range(len(fwd_log)):
                for i in range(len(actions[idx])):
                    log_p_f[idx] += fwd_log[idx, len(prev_states[idx]) + i, actions[idx][i]]
                    log_p_b[idx] += bck_log[idx, len(next_states[idx]) + i, actions[idx][i]]

        # for states, actions in trajs:
        #     for prev_state, action, next_state in zip(states[:-1], actions, states[1:]):
        #
        #         fwd_input = prev_state.copy()
        #         bck_input = next_state.copy()
        #
        #         with torch.autocast(device_type=device, dtype=torch.float16):
        #             fwd_output = policy.model(torch.tensor([fwd_input + action]).to(device))[0]
        #
        #         for t in action:
        #             with torch.autocast(device_type=device, dtype=torch.float16):
        #                 # these are the forward and backward probability estimates for each token
        #                 log_p_f[idx] += \
        #                 policy.softmax(policy.model(torch.tensor([fwd_input]).to(device))[0] / TEMPERATURE)[-1, t].log()
        #                 log_p_b[idx] += \
        #                 policy.softmax(policy.model.p_b(torch.tensor([bck_input]).to(device))[0] / TEMPERATURE)[-1, t].log()
        #
        #             fwd_input.append(t)
        #             bck_input.append(t)
        #
        #         idx += 1

        # 3. compute TB loss with TLM backward policy
        log_rewards_tensor = torch.tensor([reward for _, _, reward in trajs], device=device)

        batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)

        traj_log_p_F = scatter(log_p_f, batch_idx, dim=0, dim_size=len(traj_lens), reduce="sum")
        traj_log_p_B = scatter(log_p_b, batch_idx, dim=0, dim_size=len(traj_lens), reduce="sum")

        back_loss = traj_log_p_B.mean()
        traj_log_p_B = traj_log_p_B.detach()

        traj_diffs = (log_z + traj_log_p_F) - (log_rewards_tensor + traj_log_p_B)
        tb_loss = huber_loss(traj_diffs).mean()

        loss = tb_loss + back_loss

        tb_loss_agg += tb_loss
        back_loss_agg += back_loss

        loss.backward()

        if (r + 1) % gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()
            z_optimizer.step()
            z_optimizer.zero_grad()

            # TODO: replace this with the original logging code again
            # TODO: log mean reward and proof similarity on a set of start states
            print(tb_loss_agg, back_loss_agg)
            tb_loss_agg = back_loss_agg = 0
            print("Mean reward:", log_rewards_tensor.exp().mean())
    loop.run_until_complete(bg_loader.stop())
    loop.close()


def get_precomputed_trajectories(start_theorems: List[Theorem], tokenizer: PreTrainedTokenizerFast) -> List[
    Tuple[List[List[int]], List[List[int]]]]:
    precomputed_trajectories = []
    for thm in tqdm(start_theorems):
        states = []
        actions = []
        for tactic in thm.traced_tactics:
            new_state = tokenizer.encode(tactic.state_before)
            # TODO: substitute this with policy.build_prompt()
            states.append(new_state) if not states else states.append(
                [tokenizer.added_tokens_encoder["[STATESEP]"]] + new_state)
            actions.append(tokenizer.encode(tactic.tactic) + [tokenizer.added_tokens_encoder["[EOS]"]])
            trajectory = (states.copy(), actions.copy())
            precomputed_trajectories.append(trajectory)
    return precomputed_trajectories


def collate_skip_none(batch):
    return [i for i in batch if i is not None]


def main():
    torch.set_float32_matmul_precision('high')

    parser = ArgumentParser()
    parser.add_argument("--n-layers", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=960)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--checkpoint-path", type=str, default="model.pt")
    parser.add_argument("--reload-checkpoint", action="store_true", default=False)
    parser.add_argument("--num-tactics", type=int, default=10,
                        help="Number of tactics to sample from the policy per state")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--search-time", type=int, default=100, help="Number of MCTS nodes to explore before selecting a tactic")
    args = parser.parse_args()

    handler_factory = lambda: LeanREPLHandler(Path("./leanproject"))
    async_handler_factory = lambda: LeanREPLAsyncHandler(Path("./leanproject"))
    start_theorems = list(get_start_theorems(LEAN_DOJO_PATH / "train.json"))
    with TemporaryDirectory() as tmp_dir:
        start_states = ProofStateDataset(LEAN_DOJO_PATH / "train.json", handler_factory, Path("./mathlib4"),
                                         Path(tmp_dir))
        start_loader = DataLoader(start_states, batch_size=args.batch_size, shuffle=True, collate_fn=collate_skip_none,
                                  num_workers=args.num_workers, persistent_workers=False)

        tokenizer = PreTrainedTokenizerFast.from_pretrained("./lean_tokenizer")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        n_layers = args.n_layers
        d_model = args.d_model

        eos_id = tokenizer.added_tokens_encoder["[EOS]"]
        proofstate_id = tokenizer.added_tokens_encoder["[PROOFSTATE]"]
        proofstep_id = tokenizer.added_tokens_encoder["[PROOFSTEP]"]
        tactics_id = tokenizer.added_tokens_encoder["[TACTICS]"]
        tactics_sep_id = tokenizer.added_tokens_encoder["[SEP]"]
        # the policy sees a the list of proofstates we have transitioned to, separated by this token
        proofstate_sep_id = tokenizer.added_tokens_encoder["[STATESEP]"]

        # we need to be able to transition to unique leaf states, so end trajectories with the following tokens
        successful_proof_token = tokenizer.added_tokens_encoder["[SUC]"]
        incomplete_proof_token = tokenizer.added_tokens_encoder["[INC]"]
        invalid_proof_token = tokenizer.added_tokens_encoder["[INV]"]

        tokenizer.pad_token = "[PAD]"

        checkpoint_path = Path(args.checkpoint_path)

        # TODO: we should pretrain with the same state formulation as the GFlowNet implementation
        if args.reload_checkpoint:
            policy = MambaPolicy.from_file(checkpoint_path, True, tokenizer, device)
        else:
            config = MambaConfig(vocab_size=tokenizer.vocab_size, n_layer=n_layers, d_model=d_model)
            model = MambaLMHeadModelWrapper(config, device=device, is_gflownet=True)
            policy = MambaPolicy(model, eos_id, proofstep_id, proofstate_id, tactics_id, tactics_sep_id,
                                 proofstate_sep_id, successful_proof_token, incomplete_proof_token, invalid_proof_token,
                                 tokenizer, device, mamba_config=config)
        policy.model.train()
        # policy.model = torch.compile(policy.model)
        torch.backends.cudnn.benchmark = True

        # policy = Policy(model, eos_id, proofstep_id, proofstate_id, tactics_id, tactics_sep_id, proofstate_sep_id,
        #                 successful_proof_token, incomplete_proof_token, invalid_proof_token, tokenizer, device)

        gradient_accumulation_steps = args.gradient_accumulation_steps
        batch_size = args.batch_size
        rounds = args.rounds

        optimizer = optim.AdamW(policy.model.get_non_z_params())
        z_optimizer = optim.AdamW(policy.model.get_z_params())

        # TODO: add lr scheduler?
        precomputed_trajectories = get_precomputed_trajectories(start_theorems, tokenizer)

        train_gflownet(policy, start_loader, precomputed_trajectories, async_handler_factory, Path("./mathlib4"),
                       optimizer, z_optimizer, gradient_accumulation_steps, batch_size, 0, rounds, device,
                       max_retries=args.num_tactics, search_time=args.search_time)


if __name__ == '__main__':
    main()
