from collections.abc import Callable
from math import exp, log
import random
import time
from collections import defaultdict
from typing import Tuple, List, Union
import warnings

import numpy as np
from torch import nn
import torch
from torch_scatter import scatter
from lean_repl_py import LeanREPLHandler, LeanREPLNextProofState, LeanREPLProofState
from proofflow.model.ffm import FFM
from proofflow.data import parse_json, LEAN_DOJO_PATH, TheoremDataset, TrainSampleDataset, TrainingSample, Theorem, \
    UnknownMetaVariableError
from pathlib import Path
from torch.utils.data import DataLoader
from proofflow.policy import Policy, MambaLMHeadModelWrapper, MambaPolicy
from transformers import PreTrainedTokenizerFast
import torch.optim as optim
import wandb
from mamba_ssm.models.config_mamba import MambaConfig
from tqdm import trange, tqdm
from argparse import ArgumentParser

MAX_TRAJ_LEN = 10  # TODON: remove these and make them arguments to the file
MAX_OUTPUT_LEN = 20
GET_STATE_EVERY = 2  # we want semantically similar proofs to have the same states. Increasing this helps to do that
TEMPERATURE = 1
INVALID_REWARD = log(0.01)
INCOMPLETE_REWARD = log(0.1)


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

    def log_z(self, input_ids):
        hidden_states = self.backbone(input_ids)[:, -1]
        lm_logits = self.z_head[0](hidden_states)
        return lm_logits

    def get_non_z_params(self):
        return [i for i in self.parameters() if all(id(i) != id(j) for j in self.get_z_params())]

    def get_z_params(self):
        return self.z_head[0].parameters()


def log_reward_fn(proof_length, _compute_time):  # compute time does not work with precomputed proofs
    #log_rewards[i] = log(10) + log(1 - exp(-compute_time / 5))
    return log(10) + log(1 - exp(-proof_length / 5))

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


def sample_trajectories(
        policy: Policy,
        start_states: List[str],
        handler: LeanREPLHandler,
        envs: List[LeanREPLProofState],
        max_retries: int = 5
) -> Tuple[List[List[List[int]]], List[List[List[int]]], list[float]]:

    action_trajectories = [[] for __ in start_states]  # list of actions for each proof
    state_trajectories = [[] for __ in start_states]  # list of GFlowNet states for each proof
    proof_state_history = [[i] for i in start_states]  # list of proof states for each proof
    done = [False] * len(start_states)
    log_rewards = [INCOMPLETE_REWARD] * len(start_states)
    times = [0] * len(start_states)

    idx = 0
    while not all(done) and idx < MAX_TRAJ_LEN:

        # sample tactics

        histories, end_states = [], []

        for i, traj in enumerate(proof_state_history):
            if not done[i]:
                histories.append(traj[:-1])
                end_states.append(traj[-1])

        tactic_strings, tactic_codes, prompts = policy.next_tactics_int(end_states, max_retries, None,
                                                                        histories, temperature=1)

        # compute next states and rewards

        actions: List[Union[List[int], None]] = [None] * len(tactic_codes)
        j = 0
        for i, __ in enumerate(start_states):

            if done[i]:
                continue

            state_trajectories[i].append(prompts[i])

            for tactic_string, action in zip(tactic_strings[j], tactic_codes[j]):
                time = -time.perf_counter()
                handler.send_tactic(tactic_string, envs[i].proof_state)
                response, _ = handler.receive_json()
                time += time.perf_counter()

                has_error = "message" in response and response["message"].startswith("Lean error")
                has_error = has_error or "messages" in response and any(
                    m.severity == "error" for m in response["messages"])

                actions[j] = action
                if not has_error:  # TODO: here we just take the first one without an error. We should try something smarter like tree search
                    break

            times[i] += time

            if has_error:
                log_rewards[i] = INVALID_REWARD  # proof failed
                done[i] = True
                state_trajectories[i].append(state_trajectories[i][-1][:-1] \
                                           + [policy.proofstate_sep_id, policy.invalid_proof_token, policy.proof_step_id])
                j += 1
                continue
            assert isinstance(response, LeanREPLNextProofState)
            assert actions[j] is not None
            goals = response.goals
            action_trajectories[i] = actions[j]
            proof_state_history[i] = "\n".join(goals)

            if not goals:
                proof_length = len(actions[j])  # TODO: test some different reward formulations (e.g. computation time)
                                                # remember to change loss in 
                log_rewards[i] = log_reward_fn(proof_length, times[i])  # proof complete
                done[i] = True
                state_trajectories[i].append(state_trajectories[i][-1][:-1] \
                                           + [policy.proofstate_sep_id, policy.successful_proof_token, policy.proof_step_id])
            j += 1

        idx += 1

    for t in zip(state_trajectories, action_trajectories):
        if not done[i]:
            t[0].append([policy.incomplete_proof_token])

    return state_trajectories, action_trajectories, log_rewards


def train_gflownet(
        policy: Policy,
        start_theorems: List[Theorem],
        precomputed_trajectories: List[Tuple[List[List[int]], List[List[int]]]],
        # these are the human-written trajectories
        handler_factory: Callable[[], LeanREPLHandler],
        repo_path: Path,
        optimizer: optim.Optimizer,
        z_optimizer: optim.Optimizer,
        gradient_accumulation_steps: int,
        batch_size_replay: int,
        batch_size_sampled: int,
        rounds: int,
        eval_steps: int,
        eval_theorems: List[Theorem],
        eval_repeats: int,
        device: str,
        checkpoint_path: Path,
        metrics_path: Path,
        replay_buffer_len: int = 1_000,
        max_retries: int = 5
):
    assert precomputed_trajectories
    policy.model.train()

    optimizer.zero_grad()
    z_optimizer.zero_grad()

    replay_buffer = [None] * replay_buffer_len
    replay_end, replay_saturated = 0, False

    policy.save(checkpoint_path)

    metrics_list = []

    training_bar = trange(rounds)
    for r in training_bar:
        tb_loss_agg = back_loss_agg = 0

        with torch.no_grad():

            # 0. add new trajectories to the replay buffer

            handler = handler_factory()  # Reset the handler to avoid memory leaks

            envs = []
            start_states = []

            # sample starting states

            while len(start_states) < batch_size_replay:
                idxs = torch.randint(0, len(start_theorems), (batch_size_replay - len(start_states),))
                selected_start_thms = [start_theorems[idx] for idx in idxs]

                for i, thm in enumerate(selected_start_thms):
                    try:
                        proof_state = thm.to_proof_state(handler, repo_path=repo_path)
                    except UnknownMetaVariableError:
                        continue
                    envs.append(proof_state)
                    start_states.append(proof_state.goal)

            state_trajectories, action_trajectories, gen_log_rewards = sample_trajectories(
                policy, start_states, handler, envs, max_retries=max_retries
            )

            for t in zip(state_trajectories, action_trajectories, gen_log_rewards):

                replay_buffer[replay_end] = t
                replay_end += 1

                if replay_end >= replay_buffer_len:
                    replay_saturated = True
                    replay_end = 0

            # 1. randomly sample from the replay buffer and from human trajectories

            trajs = []

            end_idx = replay_buffer_len if replay_saturated else replay_end
            idxs_replay = torch.randint(0, end_idx, (batch_size_replay,))
            idxs_precomputed = torch.randint(0, len(precomputed_trajectories), (batch_size_sampled,))

            for i in idxs_replay:
                trajs.append(replay_buffer[i])

            for i in idxs_precomputed:
                trajs.append(precomputed_trajectories[i])

        # TODON: check this isn't always 0
        print(policy.model.z_head.grad)

        # 2. call the model on each trajectory

        starting_states = [states[0] for states, *__ in trajs]
        z_prompts = starting_states
        z_padded_prompts = policy.tokenizer.pad({"input_ids": z_prompts}, padding_side="left", return_tensors="pt")
        log_z_inputs = z_padded_prompts.input_ids.to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            log_z = policy.model.log_z(log_z_inputs)  # some duplicate computation happening here

        traj_lens = torch.tensor([len(actions) for __, actions, __ in trajs], device=device)

        # stack each trajectory so we can use torch_scatter
        log_p_f = torch.zeros(traj_lens.sum(), device=device)
        log_p_b = torch.zeros(traj_lens.sum(), device=device)

        log_rewards = []

        idx = 0
        # for each action, compute the sum of token log probs with prev state (p_f) and next state (p_b)
        for states, actions, log_reward in tqdm(trajs, leave=False):  # most of the computation is not done in this loop but bar is still useful

            log_rewards.append(log_reward)

            for prev_state, action, next_state in zip(states[:-1], actions, states[:-1]):

                fwd_input = prev_state
                bck_input = next_state

                for t in action:
                    with torch.autocast(device_type=device, dtype=torch.float16):
                        # these are the forward and backward probability estimates for each token
                        log_p_f[idx] += \
                        policy.softmax(policy.model(torch.tensor([fwd_input]).to(device))[0, -1] / TEMPERATURE)[t].log()
                        log_p_b[idx] += \
                        policy.softmax(policy.model.p_b(torch.tensor([bck_input]).to(device))[0, -1] / TEMPERATURE)[
                            t].log()

                    fwd_input += t
                    bck_input += t

                idx += 1

        # 3. compute TB loss with TLM backward policy

        log_rewards_tensor = torch.tensor(log_rewards, device=device)

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

        training_bar.set_description_str(f"tb_loss: {tb_loss:2.2f}, back_loss: {back_loss:2.2f}")
        training_bar.refresh()

        loss.backward()

        if (r + 1) % gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()
            z_optimizer.step()
            z_optimizer.zero_grad()
        
        if r % eval_steps == 0:


            with torch.no_grad():

                handler = handler_factory()

                num_theorems = 0
                envs = []
                start_states = []

                for i, thm in enumerate(eval_theorems):
                    try:
                        proof_state = thm.to_proof_state(handler, repo_path=repo_path)
                        num_theorems += 1
                        for i in eval_repeats:  # we sample each trajectory multiple times to compare similarity
                            envs.append(proof_state)
                            start_states.append(proof_state.goal)
                    except UnknownMetaVariableError:
                        warnings.warn("One of the eval theorems failed.")

                state_trajectories, action_trajectories, gen_log_rewards = sample_trajectories(
                    policy, start_states, handler, max_retries=max_retries
                )

                mean_similarity = 0
                for i in range(num_theorems):
                    trajs = action_trajectories[i*eval_repeats:(i+1)*eval_repeats]
                    num_unique = len(set(trajs))  # TODO: we should use a better similarity metric
                    mean_similarity += num_unique / num_theorems

            metrics = {
                "sampled_mean_reward": log_rewards_tensor.mean().item(),
                "eval_mean_reward": gen_log_rewards.mean().item(),
                "eval_similarity": mean_similarity,
                "tb_loss": tb_loss_agg,
                "back_loss": back_loss_agg
            }
            metrics_list.append(metrics)

            np.save(metrics_path, np.array(metrics_list, dtype=object), allow_pickle=True)

            policy.save(checkpoint_path)

            #wandb.log(metrics, step=r)
            #wandb.log_model(checkpoint_path, name=f"model-round-{r}")

def get_precomputed_trajectories(start_theorems: List[Theorem], tokenizer: PreTrainedTokenizerFast, policy: Policy) -> List[Tuple[List[List[int]], List[List[int]], int]]:

    precomputed_trajectories = []
    for thm in tqdm(start_theorems):
        gfn_actions = []
        gfn_states = []
        proof_states = []
        for tactic in thm.traced_tactics:
            gfn_actions.append(tokenizer.encode(tactic.tactic))
            gfn_states.append(policy._build_prompt(tactic.state_before, None, proof_states))
            proof_states.append(tactic.state_before)
        gfn_states.append(policy._build_prompt([policy.successful_proof_token], None, proof_states))
        log_reward = log_reward_fn(len(gfn_actions), 0) if thm.traced_tactics[-1].state_after == "no goals" else INCOMPLETE_REWARD
        precomputed_trajectories.append(gfn_states, gfn_actions, log_reward)

def main():
    torch.set_float32_matmul_precision('high')

    parser = ArgumentParser()
    parser.add_argument("--n-layers", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=960)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--load-checkpoint-path", type=str, default="model.pt")
    parser.add_argument("--save-checkpoint-path", type=str, default="checkpoint.pt")
    parser.add_argument("--save-metrics-path", type=str, default="metrics.npy")
    parser.add_argument("--reload-checkpoint", action="store_true", default=False)
    parser.add_argument("--eval-steps", type=int, default=1)
    parser.add_argument("--eval-theorems", type=int, default=1)
    args = parser.parse_args()

    handler_factory = lambda: LeanREPLHandler(Path("./leanproject"))
    start_theorems = list(get_start_theorems(LEAN_DOJO_PATH / "train.json"))

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

    if args.reload_checkpoint:
        policy = MambaPolicy.from_file(checkpoint_path, True, tokenizer, device)
    else:
        config = MambaConfig(vocab_size=tokenizer.vocab_size, n_layer=n_layers, d_model=d_model)
        model = MambaLMHeadModelWrapper(config, device=device, is_gflownet=True)
        policy = MambaPolicy(model, eos_id, proofstep_id, proofstate_id, tactics_id, tactics_sep_id,
                             proofstate_sep_id, successful_proof_token, incomplete_proof_token, invalid_proof_token,
                             tokenizer, device, mamba_config=config)
    policy.model.train()
    #policy.model = torch.compile(policy.model)
    torch.backends.cudnn.benchmark = True

    # policy = Policy(model, eos_id, proofstep_id, proofstate_id, tactics_id, tactics_sep_id, proofstate_sep_id,
    #                 successful_proof_token, incomplete_proof_token, invalid_proof_token, tokenizer, device)

    gradient_accumulation_steps = args.gradient_accumulation_steps
    batch_size = args.batch_size
    rounds = args.rounds
    eval_steps = args.eval_steps

    eval_theorems = start_theorems[:args.eval_theorems]

    optimizer = optim.AdamW(policy.model.get_non_z_params())
    z_optimizer = optim.AdamW(policy.model.get_z_params())

    precomputed_trajectories = get_precomputed_trajectories(start_theorems, tokenizer, policy)

    config = {"n_layers": n_layers, "d_model": d_model, "rounds": rounds, "batch_size": batch_size,
              "gradient_accumulation_steps": gradient_accumulation_steps}
    #wandb.init(project="proofflow", config=config)
    train_gflownet(policy, start_theorems, precomputed_trajectories, handler_factory, Path("./mathlib4"), optimizer,
                   z_optimizer, gradient_accumulation_steps, batch_size, batch_size, rounds, eval_steps, eval_theorems,
                   5, device, Path(args.save_checkpoint_path), Path(args.save_metrics_path))
    #wandb.finish(exit_code=0)


if __name__ == '__main__':
    main()
