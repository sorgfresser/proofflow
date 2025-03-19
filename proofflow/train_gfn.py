import gc
from dataclasses import dataclass
from math import exp, log, ceil, floor
import time
from typing import Tuple, List, Union, Dict, Any, Iterator, Optional, Callable
from functools import partial
from torch import nn
import torch
from torch_scatter import scatter
from lean_repl_py import LeanREPLHandler, LeanREPLNextProofState, LeanREPLProofState, LeanREPLAsyncHandler
from proofflow.model.ffm import FFM
from proofflow.data import parse_json, LEAN_DOJO_PATH, Theorem, ProofStateDataset, TheoremDataset
from pathlib import Path
from tempfile import TemporaryDirectory
from torch.utils.data import DataLoader
from proofflow.policy import Policy, MambaLMHeadModelWrapper, MambaPolicy
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
import torch.optim as optim
from math import sqrt
from mamba_ssm.models.config_mamba import MambaConfig
from tqdm import trange, tqdm
from argparse import ArgumentParser
import wandb
import numpy as np
import random
from transformers import AutoModel, AutoTokenizer
import re


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
                # Avoid cycles
                if proof_states[idx] in self.previous_states:
                    continue
                self.children_for_tactics[tactic] = Node(proof_states[idx], parent=self, parent_tactic=tactic,
                                                         previous_states=self.previous_states + [self.proof_state],
                                                         proof_state_idx=indices[idx], metadata=self.metadata)
                self.times[tactic] = times[idx]
                self.total_action_values[tactic] = values[idx]
                self.visit_counts[tactic] = 1
        if not self.children_for_tactics:
            self.backup_branch_done()
            return
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

    def subtree_count(self):
        return 1 + sum(child.subtree_count() for child in self.children_for_tactics.values())

    def valid_count(self):
        if self.branch_is_done:
            return 0
        return 1 + sum(child.valid_count() for child in self.children_for_tactics.values())



class MCTS:
    def __init__(self, root: Node):
        self.root = root
        self.metadata = root.metadata
        self.proof = ""
        self.time = 0.0
        self.step_count = 0
        self.done = False
        self.last_tactic = ""
        self.solved = False

    def move(self):
        if self.done:
            return
        best_tactic = self.root.best_action()
        self.proof += best_tactic + "\n"
        self.time += self.root.times[best_tactic]
        self.root = self.root.children_for_tactics[best_tactic]
        self.root.parent = None  # speedup memory release
        self.step_count += 1
        self.last_tactic = best_tactic
        if self.root.branch_is_done:
            self.done = True
        gc.collect()

    def select(self) -> Node:
        node = self.root
        while node != node.select():
            node = node.select()
        if node.branch_is_done:
            self.done = True
        return node


def _get_start_states(start_loader: Iterator, handlers: list[LeanREPLHandler], num_repeats: int = 1) -> \
        Optional[Tuple[List[LeanREPLProofState], List[MCTS]]]:
    try:
        batch = next(start_loader)
    except StopIteration:
        return None
    thms, paths = [elem[0] for elem in batch], [elem[1] for elem in batch]
    # Will be thm[0] k times, then thm 1 k times etc.
    thms = [thm for thm in thms for _ in range(num_repeats)]
    paths = [path for path in paths for _ in range(num_repeats)]
    # Start all REPLs
    proof_states= [handler.unpickle_proof_state(path) for handler, path in zip(handlers, paths)]
    # Gather
    proof_states = [proof_state for proof_state, env in proof_states]
    # Unlink them all, not needed anymore
    assert all(path.exists() for path in paths)
    for path in paths:
        path.unlink(missing_ok=True) # missing ok because of repeats
    assert all(len(proof_state.goals) == 1 for proof_state in proof_states)
    nodes = [Node(proof_state.goals[0], proof_state_idx=proof_state.proof_state,
                  metadata={"theoremname": thm.full_name, "theoremfile": thm.file_path}) for proof_state, thm in
             zip(proof_states, thms)]
    return proof_states, [MCTS(node) for node in nodes]


period_regex = re.compile(r"(\w+)(\s)\.(\s)(\w+)")
subscript_regex = re.compile(r"(\w+)(\s)([₀₁₂₃₄₅₆₇₈₉])")

def postprocess_tactic(tactic: str):
    original = tactic
    tactic = re.sub(period_regex, r"\g<1>.\g<4>", original)
    tactic = re.sub(subscript_regex, r"\g<1>\g<3>", tactic)
    while original != tactic:
        original = tactic
        tactic = re.sub(period_regex, r"\g<1>.\g<4>", original)
        tactic = re.sub(subscript_regex, r"\g<1>\g<3>", tactic)
    return tactic


def _envs_expand(
        handlers: list[LeanREPLHandler],
        tactics: list[list[str]],
        proof_state_indices: list[list[int]]
):
    """Sends one tactic at a time per handler"""
    assert len(handlers) == len(tactics) == len(proof_state_indices)
    proven = [[] for _ in handlers]
    invalid = [[] for _ in handlers]
    indices = [[] for _ in handlers]
    goals = [[] for _ in handlers]
    times = [[] for _ in handlers]
    for i in range(len(handlers)):
        for tactic, proof_state_idx in zip(tactics[i], proof_state_indices[i], strict=True):
            handler = handlers[i]
            start_time = time.perf_counter()
            handler.send_tactic(tactic, proof_state_idx)
            response, _ = handler.receive_json()
            times[i].append(time.perf_counter() - start_time)
            has_error = "message" in response and response["message"].startswith("Lean error")
            has_error = has_error or "messages" in response and any(
                msg.severity == "error" for msg in response["messages"])
            has_error = has_error or (isinstance(response, LeanREPLNextProofState) and any(
                msg.severity == "error" for msg in response.messages))
            # Try again with the postprocessed version
            if has_error:
                start_time = time.perf_counter()
                handler.send_tactic(postprocess_tactic(tactic), proof_state_idx)
                response, _ = handler.receive_json()
                times[i][-1] = time.perf_counter() - start_time
                has_error = "message" in response and response["message"].startswith("Lean error")
                has_error = has_error or "messages" in response and any(
                    msg.severity == "error" for msg in response["messages"])
                has_error = has_error or (isinstance(response, LeanREPLNextProofState) and any(
                    msg.severity == "error" for msg in response.messages))
            if has_error:
                invalid[i].append(True)
                proven[i].append(False)
                indices[i].append(None)
                goals[i].append(None)
                continue
            assert isinstance(response, LeanREPLNextProofState)
            proven[i].append(not response.goals)
            invalid[i].append(False)
            if not response.goals:
                goals[i].append(None)
            else:
                goals[i].append(response.goals[0])  # only need one goal, it has all information
            indices[i].append(response.proof_state)
    return proven, invalid, indices, goals, times


def _compute_log_rewards(proven: List[bool], invalid: List[bool], times: List[float], length: int) -> List[float]:
    rewards = []
    for p, i, _t in zip(proven, invalid, times):
        if i:
            rewards.append(log(0.01))
        elif p:  # proof complete
            #rewards.append(1 + 15*exp(-t))  # compute time does not work with precomputed proofs
            rewards.append(log(10) + log(1 - exp(-length / 5)))
        else:  # ongoing = small reward
            #rewards.append(0.1 + 0.25*exp(-t))  # compute time does not work with precomputed proofs
            rewards.append(log(0.1) + log(1 - exp(-length / 5)))
    return rewards

critic_model = AutoModel.from_pretrained(
    "internlm/internlm2_5-step-prover-critic",
    device_map="cpu",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
critic_tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-step-prover-critic", trust_remote_code=True)


def _compute_log_internlm(proven: List[bool], invalid: List[bool], times: List[float], length: int, proof_states: List[str]) -> List[float]:
    rewards = []
    for p, i, _t, proof_state in zip(proven, invalid, times, proof_states):
        chat = [
            {"role": "user", "content": "Which state is closer to 'no goals'?"},
            {"role": "assistant", "content": f"{proof_state}"},
        ]
        if i:
            rewards.append(log(0.01))
        elif p:  # proof complete
            # rewards.append(1 + 15*exp(-t))  # compute time does not work with precomputed proofs
            rewards.append(log(10) + log(1 - exp(-length / 5)))
        else:  # ongoing = small reward
            # rewards.append(0.1 + 0.25*exp(-t))  # compute time does not work with precomputed proofs
            score1 = critic_model.get_score(critic_tokenizer, chat)
            rewards.append(log(score1 + 4) + log(1 - exp(-length / 5)))
    return rewards


def sample_mcts_trajectories(
        policy: Policy,
        start_states: List[MCTS],
        handlers: List[LeanREPLHandler],
        search_time: int,
        device: str,
        max_len: int = 10,
        max_retries: int = 5,
        temperature: float = 0.7
) -> Tuple[List[List[List[int]]], List[List[List[int]]], list[float], int, float]:
    action_trajectories = [[] for __ in start_states]  # list of actions for each proof
    state_trajectories = [[] for __ in start_states]  # list of GFlowNet states for each proof
    done = [False] * len(start_states)
    log_rewards = [0] * len(start_states)

    idx = 0
    nodes_proven = 0
    expanded_node_count = 0
    valid_child_count = 0
    while not all(node.done for node in start_states) and idx < max_len:

        try:

            for _ in range(search_time):

                if all(node.done for node in start_states):
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

                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    tactic_strings = policy.next_tactics(end_states, max_retries, None, histories, temperature=temperature)

                current_handlers = [handlers[node_idx] for node_idx in range(len(start_states)) if not start_states[node_idx].done]
                proven, invalid, indices, goals, times_current = _envs_expand(current_handlers, tactic_strings,
                                            [[current.proof_state_idx] * len(tactic_strings[current_idx]) for current_idx, current in enumerate(currents)],)
                current_idx = 0

                for i, node in enumerate(start_states):

                    if node.done:
                        continue

                    rewards = _compute_log_rewards(proven[current_idx], invalid[current_idx], times_current[current_idx], len(currents[current_idx].previous_states) + 1)
                    # Only passes on valid tactics to expand, we might want to change this

                    tactics = [t for t, p in zip(tactic_strings[current_idx], invalid[current_idx], strict=True) if not p]
                    goals_node = [g for g, p in zip(goals[current_idx], invalid[current_idx], strict=True) if not p]
                    times_current_node = [t for t, p in zip(times_current[current_idx], invalid[current_idx], strict=True) if not p]
                    indices_node = [index for index, p in zip(indices[current_idx], invalid[current_idx], strict=True) if not p]
                    rewards = [r for r, p in zip(rewards, invalid[current_idx], strict=True) if not p]
                    expanded_node_count += 1
                    if tactics:
                        valid_child_count += 1
                    currents[current_idx].expand(tactics, goals_node, times_current_node, rewards, indices_node)
                    if any(proven[current_idx]):
                        node.done = True
                        node.solved = True
                        node.last_tactic = tactic_strings[current_idx][proven[current_idx].index(True)]
                        nodes_proven += 1
                    # Edge case, if we only have invalid tactics, there is no way to continue
                    elif node.root.branch_is_done:
                        node.done = True
                        node.solved = False
                        node.last_tactic = tactic_strings[current_idx][0] # all of them are unproven, just take 0
                    current_idx += 1
        except BrokenPipeError as e:
            raise e
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
            failed_node = current
            while failed_node.parent is not None:
                strings.append(failed_node.parent_tactic)
                failed_node = failed_node.parent
            current_proof = "\n".join(reversed(strings))
            print(f"Current proof: {current_proof}")
            raise e

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
            if node.done:
                end_token = policy.successful_proof_token if node.solved else policy.invalid_proof_token
                state_trajectories[i].append(prompts[i][:-1] + [policy.proofstate_sep_id, end_token, policy.proof_step_id])
                done[i] = True

        idx += 1
    for i, node in enumerate(start_states):
        log_rewards[i] = _compute_log_internlm([node.solved], [not node.solved and node.done], [node.time], node.step_count + 1, [node.root.proof_state])[0]
    for i, t in enumerate(state_trajectories):
        if not start_states[i].done:
            t.append(t[-1][:-1] + [policy.proofstate_sep_id, policy.incomplete_proof_token, policy.proof_step_id])
    assert all(len(state) == len(action) + 1 for state, action in zip(state_trajectories, action_trajectories, strict=True))

    return state_trajectories, action_trajectories, log_rewards, nodes_proven, float(valid_child_count) / expanded_node_count



# TODO: use levenshtein distance
def get_similarity(action_trajs: List[List[List[int]]], N: int = 2) -> float:
    return 0  # TODO: fix list not hashable
    # here we compute the mean number of length N subsequences in pairs of trajectories
    result = 0

    traj_dicts = []
    for action_traj in action_trajs:
        traj_dict = {}
        for sub_idx in range(len(action_traj)-N+1):
            sub_seq = action_traj[sub_idx:sub_idx+N]
            traj_dict[sub_seq] = traj_dict.get(sub_seq, 0) + 1
        traj_dicts.append(traj_dict)

    for a_0 in traj_dicts:
        for a_1 in traj_dicts:
            result += [a_1.get(k, 0) * v for k, v in a_0.items()]
    return result / len(action_trajs)**2


class PrecomputedTrajectoryDataset(TheoremDataset):
    def __init__(self, data_path: Path, tokenizer: PreTrainedTokenizer, policy: Policy, create_lookup: bool = False):
        super().__init__(data_path)
        self.tokenizer = tokenizer
        self.policy = policy
        self._lookup = None
        if create_lookup:
            self._lookup = {thm.full_name: idx for idx, thm in enumerate(self.thms)}

    def __getitem__(self, item: int) -> Tuple[List[List[int]], List[List[int]], float]:
        thm = super().__getitem__(item)
        gfn_actions: List[List[int]] = []
        gfn_states: List[List[int]] = []
        proof_states: List[str] = []
        for tactic in thm.traced_tactics:
            gfn_actions.append(self.tokenizer.encode(tactic.tactic))
            gfn_states.append(self.policy._build_prompt(tactic.state_before, None, proof_states))
            proof_states.append(tactic.state_before)
        gfn_states.append(
            self.policy._build_prompt(self.tokenizer.decode([self.policy.successful_proof_token]), None, proof_states))
        complete = thm.traced_tactics[-1].state_after == "no goals"
        log_reward: float = _compute_log_rewards([complete], [not complete], [0], len(gfn_actions))[0]
        return gfn_states, gfn_actions, log_reward

    def lookup(self, theorem_name: str) -> int:
        assert self._lookup
        return self._lookup[theorem_name]



def evaluate(policy: Policy, eval_loader: DataLoader, handler_factory: Callable[[], LeanREPLHandler],
             device: str, eval_repeats: int, search_time: int, max_retries: int, batch_size: int, name: str) -> dict[str, float]:

    similarities = []
    rewards = []
    handlers = [handler_factory() for _ in range(batch_size)]
    eval_iter = iter(eval_loader)
    with torch.no_grad():
        batch = _get_start_states(eval_iter, handlers)
        with tqdm(total=len(eval_loader)) as pbar:
            while batch is not None:
                # If the whole batch was invalid, quite unlikely, but possible
                if not batch:
                    batch = _get_start_states(eval_iter, handlers)
                    continue
                _, start_states = batch
                try:
                    _state_trajectories, action_trajectories, gen_log_rewards, nodes_proven, proof_state_ratio = sample_mcts_trajectories(
                        policy, start_states, handlers, search_time, device, max_retries=max_retries
                    )
                except BrokenPipeError:
                    print("Broken pipe error, starting eval all over")
                    handlers = [handler_factory() for _ in range(batch_size)]
                    eval_iter = iter(eval_loader)
                    batch = _get_start_states(eval_iter, handlers)
                    continue
                mean_similarity = 0
                for i in range(len(action_trajectories) // eval_repeats):
                    trajs = action_trajectories[i * eval_repeats:(i + 1) * eval_repeats]
                    mean_similarity += get_similarity(trajs)

                gen_log_rewards_tensor = torch.tensor(gen_log_rewards, device=device)
                similarities.append(mean_similarity)
                rewards.append(gen_log_rewards_tensor.exp().mean().item())
                batch = _get_start_states(eval_iter, handlers)
                pbar.update(1)

    metrics = {
        f"validation/mean_reward_{name}": sum(rewards) / len(rewards),
        f"validation/similarity_{name}": float(sum(similarities)) / len(similarities),
        f"validation/nodes_proven_{name}": nodes_proven,
        f"validation/proof_state_ratio_{name}": proof_state_ratio
    }
    return metrics

def linear_schedule_length(round, initial_length: int = 1, every_steps: int = 1) -> int:
    steps = round // every_steps
    return steps + initial_length


def train_gflownet(
        policy: Policy,
        start_loader: DataLoader,
        precomputed_trajectories: PrecomputedTrajectoryDataset,
        # these are the human-written trajectories
        handler_factory: Callable[[], LeanREPLHandler],
        optimizer: optim.Optimizer,
        z_optimizer: optim.Optimizer,
        gradient_accumulation_steps: int,
        batch_size_replay: int,
        batch_size_sampled: int,
        rounds: int,
        eval_steps: int,
        eval_loader: DataLoader,
        eval_batch_size: int,
        eval_repeats: int,
        device: str,
        checkpoint_path: Path,
        metrics_path: Path,
        length_schedule: Callable[[int], int],
        replay_buffer_len: int = 1_000,
        max_retries: int = 5,
        search_time: int = 100,
        train_repeats: int = 1,
        temperature: float = 0.7
):

    policy.model.train()
    policy.save(checkpoint_path)

    optimizer.zero_grad()
    z_optimizer.zero_grad()

    scaler = torch.GradScaler()

    replay_buffer = [None] * replay_buffer_len
    replay_end, replay_saturated = 0, False

    tb_loss_acc = back_loss_acc = reward_acc = 0
    metrics_list = []

    print("Training...")
    training_bar = trange(rounds)

    data_iter = iter(start_loader)
    for r in training_bar:
        handlers = [handler_factory() for _ in range(batch_size_replay * train_repeats)]
        with torch.no_grad():
            # 0. add new trajectories to the replay buffer
            current_length = length_schedule(r)
            batch = _get_start_states(data_iter, handlers, num_repeats=train_repeats)
            if batch is None:
                data_iter = iter(start_loader)
                batch = _get_start_states(data_iter, handlers, num_repeats=train_repeats)
            # If the whole batch was invalid, quite unlikely, but possible
            if not batch:
                continue
            _, start_states = batch
            try:
                state_trajectories, action_trajectories, gen_log_rewards, nodes_proven, proof_state_ration = sample_mcts_trajectories(
                    policy, start_states, handlers, search_time, device, max_retries=max_retries, max_len=current_length, temperature=temperature
                )
            except BrokenPipeError:
                # Reset handlers, try again
                print("BrokenPipeError, resetting handlers!")
                continue
            gc.collect()

            # for t in zip(state_trajectories, action_trajectories, gen_log_rewards):
            #
            #     replay_buffer[replay_end] = t
            #     replay_end += 1
            #
            #     if replay_end >= replay_buffer_len:
            #         replay_saturated = True
            #         replay_end = 0

            sampled_mean_reward = sum(map(exp, gen_log_rewards)) / len(gen_log_rewards)
            action_lengths = sum(map(len, action_trajectories)) / len(action_trajectories)

            # 1. randomly sample from the replay buffer and from human trajectories
            # end_idx = replay_buffer_len if replay_saturated else replay_end
            # idxs_replay = torch.randint(0, end_idx, (batch_size_replay,))
            # idxs_precomputed = torch.randint(0, len(precomputed_trajectories), (batch_size_sampled,))
            # trajs = [replay_buffer[i] for i in idxs_replay] + [precomputed_trajectories[i] for i in idxs_precomputed]
            idxs_precomputed = set([precomputed_trajectories.lookup(state.metadata["theoremname"]) for state in start_states])
            trajs = list(zip(state_trajectories, action_trajectories, gen_log_rewards)) + [precomputed_trajectories[i] for i in idxs_precomputed]
        # 2. call the model on each trajectory

        z_prompts = [states[0] for states, _, _ in trajs]
        z_padded_prompts = policy.tokenizer.pad({"input_ids": z_prompts}, padding_side="right", return_tensors="pt")
        log_z_inputs = z_padded_prompts.input_ids.to(device)
        log_z_attention_mask = z_padded_prompts.attention_mask.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            log_z = policy.model.log_z(log_z_inputs, log_z_attention_mask)  # some duplicate computation happening here

            traj_lens = torch.tensor([len(actions) for __, actions, __ in trajs], device=device)

            # stack each trajectory so we can use torch_scatter
            log_p_f = torch.zeros(traj_lens.sum(), device=device)
            log_p_b = torch.zeros(traj_lens.sum(), device=device)

            idx = 0
            # for each action, compute the sum of token log probs with prev state (p_f) and next state (p_b)
            prev_states = [state for states, __, __ in trajs for state in states[:-1]]
            next_states = [state for states, __, __ in trajs for state in states[1:]]
            actions = [action for __, actions, __ in trajs for action in actions]
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
            for idx in range(len(actions)):
                for i in range(len(actions[idx])):
                    log_p_f[idx] += fwd_log[idx, len(prev_states[idx]) + i, actions[idx][i]]
                    log_p_b[idx] += bck_log[idx, len(next_states[idx]) + i, actions[idx][i]]

            # 3. compute TB loss with TLM backward policy
            log_rewards_tensor = torch.tensor([reward for _, _, reward in trajs], device=device)

            batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)

            traj_log_p_F = scatter(log_p_f, batch_idx, dim=0, dim_size=len(traj_lens), reduce="sum")
            traj_log_p_B = scatter(log_p_b, batch_idx, dim=0, dim_size=len(traj_lens), reduce="sum")

            back_loss = -traj_log_p_B.mean()
            traj_log_p_B = traj_log_p_B.detach()

            traj_diffs = (log_z + traj_log_p_F) - (log_rewards_tensor + traj_log_p_B)
            tb_loss = huber_loss(traj_diffs).mean()

            loss = tb_loss + back_loss

            tb_loss_acc += tb_loss.item()
            back_loss_acc += back_loss.item()

            training_bar.set_description_str(f"tb_loss: {tb_loss:2.2f}, back_loss: {back_loss:2.2f}")
            training_bar.refresh()

        unique_trajs = len(set([tuple([tuple(action) for action in actions]) for __, actions, __ in trajs]))
        scaler.scale(loss).backward()
        reward_acc += sampled_mean_reward / gradient_accumulation_steps
        training_metrics = {"train/tb_loss": tb_loss.item(), "train/back_loss": back_loss.item(),
                            "train/sampled_mean_reward": sampled_mean_reward,
                            "train/mean_action_p_f": log_p_f.mean().item(), "train/mean_action_p_b": log_p_b.mean().item(),
                            "train/mean_log_z": log_z.mean().item(), "train/std_log_z": log_z.std().item(),
                            "train/action_lengths": action_lengths, "train/mcts_max_length": current_length, 
                            "train/nodes_proven": nodes_proven, "train/proof_state_ration": proof_state_ration,
                            "train/unique": unique_trajs/len(trajs)}
        wandb.log(training_metrics, step=r)

        if (r + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            scaler.unscale_(z_optimizer)
            nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.step(z_optimizer)
            scaler.update()
            optimizer.zero_grad()
            z_optimizer.zero_grad()

            training_metrics.update({"train/accumulated_tb_loss": tb_loss_acc,
                                     "train/accumulated_back_loss": back_loss_acc,
                                     "train/reward_acc": reward_acc})
            tb_loss_acc = back_loss_acc = reward_acc = 0

        wandb.log(training_metrics, step=r)

        if r % eval_steps == 0:
            eval_metrics = evaluate(policy, eval_loader, handler_factory, device, eval_repeats, search_time, max_retries, eval_batch_size, "a")
            eval_metrics = evaluate(policy, eval_loader, handler_factory, device, eval_repeats, search_time, max_retries, eval_batch_size, "b")
            eval_metrics.update(training_metrics)
            metrics_list.append(eval_metrics)
            np.save(metrics_path, np.array(metrics_list, dtype=object), allow_pickle=True)
            policy.save(checkpoint_path)
            wandb.log(eval_metrics, step=r)
            wandb.log_model(checkpoint_path, name=f"model-round-{r}")

def collate_skip_none(batch):
    return [i for i in batch if i is not None]

def main():
    torch.set_float32_matmul_precision('high')

    parser = ArgumentParser()
    parser.add_argument("--n-layers", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=960)
    parser.add_argument("--rounds", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--load-checkpoint-path", type=str, default="model.pt")
    parser.add_argument("--save-checkpoint-path", type=str, default="checkpoint.pt")
    parser.add_argument("--save-metrics-path", type=str, default="metrics.npy")
    parser.add_argument("--reload-checkpoint", action="store_true", default=True)
    parser.add_argument("--num-tactics", type=int, default=100,
                        help="Number of tactics to sample from the policy per state")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--search-time", type=int, default=5, help="Number of MCTS nodes to explore before selecting a tactic")
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--eval-theorems", type=int, default=20)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--train-repeats", type=int, default=127)
    parser.add_argument("--train-on-eval", action="store_true", default=True)
    parser.add_argument("--batch-size-precomputed", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    product = args.batch_size * (args.train_repeats + 1)
    if product > args.max_batch_size:
        batch_size = floor(args.max_batch_size / (args.train_repeats + 1))
        assert batch_size > 0
        gradient_accumulation_steps_mult = ceil(args.batch_size / batch_size)
    else:
        batch_size = args.batch_size
        gradient_accumulation_steps_mult = 1

    handler_factory = lambda: LeanREPLHandler(Path("./leanproject"))

    with TemporaryDirectory() as tmp_dir:
        print("Getting data...")
        start_states = ProofStateDataset(LEAN_DOJO_PATH / "train.json", handler_factory, Path("./mathlib4"),
                                         Path(tmp_dir))
        start_loader = DataLoader(start_states, batch_size=batch_size, shuffle=True, collate_fn=collate_skip_none,
                                  num_workers=args.num_workers, persistent_workers=False)
        eval_states = ProofStateDataset(Path("./proof_flow_theorems.json"), handler_factory, Path("./mathlib4"), Path(tmp_dir), repeats=args.eval_repeats, sample_count=args.eval_theorems, filter_lake=False)
        if args.train_on_eval:
            start_loader = DataLoader(eval_states, batch_size=batch_size, shuffle=True, collate_fn=collate_skip_none,
                                     num_workers=args.num_workers, persistent_workers=False)
        eval_loader = DataLoader(eval_states, batch_size=args.eval_batch_size,
                                 collate_fn=collate_skip_none, shuffle=False, num_workers=args.num_workers)
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

        checkpoint_path = Path(args.load_checkpoint_path)
        print("Initializing model")
        if args.reload_checkpoint:
            policy = MambaPolicy.from_file(checkpoint_path, True, tokenizer, device)
            d_model = policy.config.d_model
            n_layers = policy.config.n_layer
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

        gradient_accumulation_steps = args.gradient_accumulation_steps * gradient_accumulation_steps_mult
        rounds = args.rounds * args.gradient_accumulation_steps
        eval_steps = args.eval_steps
        eval_repeats = args.eval_repeats
        eval_batch_size = args.eval_batch_size
        train_repeats = args.train_repeats

        optimizer = optim.AdamW(policy.model.get_non_z_params(), lr=args.lr)
        z_optimizer = optim.AdamW(policy.model.get_z_params(), lr=1e-3)

        print("Initializing precomputed trajectories")
        if args.train_on_eval:
            precomputed_trajectories = PrecomputedTrajectoryDataset(Path("./proof_flow_theorems.json"), tokenizer, policy, create_lookup=True)
        else:
            precomputed_trajectories = PrecomputedTrajectoryDataset(LEAN_DOJO_PATH / "train.json", tokenizer, policy, create_lookup=True)

        config = {"n_layers": n_layers, "d_model": d_model, "rounds": rounds, "batch_size": batch_size,
                  "gradient_accumulation_steps": gradient_accumulation_steps, "eval_steps": eval_steps,
                  "eval_repeats": eval_repeats, "eval_theorems": args.eval_theorems, "num_tactics": args.num_tactics,
                  "search_time": args.search_time, "eval_batch_size": eval_batch_size, "seed": seed, "train_repeats": train_repeats}
        wandb.init(project="proofflow", config=config, entity="scalogi")
        save_checkpoint_path = Path(args.save_checkpoint_path + "_" + str(batch_size) + "_" + str(train_repeats))
        save_metrics_path = Path(args.save_metrics_path + "_" + str(batch_size) + "_" + str(train_repeats))
        train_gflownet(policy, start_loader, precomputed_trajectories, handler_factory, optimizer, z_optimizer,
                       gradient_accumulation_steps, batch_size, args.batch_size_precomputed, rounds, eval_steps, eval_loader,
                       eval_batch_size, eval_repeats, device, save_checkpoint_path, save_metrics_path,
                       partial(linear_schedule_length, initial_length=1, every_steps=100),
                       max_retries=args.num_tactics, search_time=args.search_time, train_repeats=train_repeats, temperature=args.temperature)

        wandb.finish(exit_code=0)


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore", message="^.*using the `__call__` method is faster than.*$")

    main()
