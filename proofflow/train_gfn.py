from math import log
from collections import defaultdict
from typing import Tuple, List

from torch import nn
import torch
from torch_scatter import scatter
from lean_repl_py import LeanREPLHandler, LeanREPLNextProofState
from proofflow.model.ffm import FFM
from proofflow.data import parse_json, LEAN_DOJO_PATH, TheoremDataset, TrainSampleDataset, TrainingSample, Theorem, UnknownMetaVariableError
from pathlib import Path
from torch.utils.data import DataLoader
from proofflow.policy import Policy, MambaLMHeadModelWrapper, MambaPolicy
from transformers import PreTrainedTokenizerFast
import torch.optim as optim
import wandb
from mamba_ssm.models.config_mamba import MambaConfig
from tqdm import tqdm
from argparse import ArgumentParser


MAX_TRAJ_LEN = 10
MAX_OUTPUT_LEN = 20
GET_STATE_EVERY = 1  # we want semantically similar proofs to have the same states. Increasing this helps to do that
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
        self.z_head = (nn.Linear(20, 1, bias=True), ) # hack to not register as parameter
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
        return self.parameters()

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
        yield thm


def train_gflownet(
        policy: Policy,
        start_theorems: List[Theorem],
        precomputed_trajectories: List[Tuple[List[List[int]], List[List[int]]]],  # these are the human-written trajectories
        handler: LeanREPLHandler,
        repo_path: Path,
        optimizer: optim.Optimizer,
        z_optimizer: optim.Optimizer,
        gradient_accumulation_steps: int,
        batch_size_replay: int,
        batch_size_sampled: int,
        rounds: int,
        device: str,
        replay_buffer_len: int = 1_000
    ):
    assert precomputed_trajectories
    policy.model.train()

    optimizer.zero_grad()
    z_optimizer.zero_grad()

    replay_buffer = [None] * replay_buffer_len
    replay_end, replay_saturated = 0, False

    tb_loss_agg = back_loss_agg = 0

    for r in range(rounds):

        with torch.no_grad():

            # 0. add new trajectories to the replay buffer
            envs = []
            start_states = []

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

            action_trajectories = [[] for __ in start_states]  # list of actions for each proof
            state_trajectories = [[] for i in start_states]  # list of GFlowNet states for each proof
            proof_state_history = [[i] for i in start_states]  # list of proof states for each proof
            done = [False] * len(start_states)
            log_rewards = [log(0.1)] * len(start_states)  # an inprogress trajectory gets more reward than a failed one
                                                          # TODO: we should reconsider the reward setup here

            idx = 0
            while not all(done) and idx < MAX_TRAJ_LEN:

                #prompts = [
                #      [policy.proof_state_id] \
                #    + [
                #        t for s in state[(len(state)-1) % GET_STATE_EVERY :: GET_STATE_EVERY]
                #          for t in policy.tokenizer.encode(s) + [policy.proofstate_sep_id]
                #      ][:-1] \
                #    + [policy.proof_step_id]
                #        for state in state_trajectory
                #]

                prompts = []
                for i, traj in enumerate(proof_state_history):

                    if done[i]:
                        continue

                    prompt = [policy.tokenizer.encode(s) for s in traj[(len(traj)-1) % GET_STATE_EVERY :: GET_STATE_EVERY]]
                    prompt = [t for s in prompt for t in s + [policy.proofstate_sep_id]][:-1]
                    prompt = [policy.proof_state_id] + prompt + [policy.proof_step_id]
                    prompts.append(prompt)

                    state_trajectories[i].append(prompt)

                padded = policy.tokenizer.pad({"input_ids": prompts}, padding_side="left", return_tensors="pt") # TODO: pad right instead most likely
                inputs = padded.input_ids.to(device)

                actions = [[] for __ in inputs]
                eos = [False] * len(inputs)

                output_idx = 0
                while not all(eos) and output_idx < MAX_OUTPUT_LEN:

                    with torch.autocast(device_type=device, dtype=torch.float16):
                        logits = policy.model(inputs)[:, -1, ...]  # TODO: wasteful to compute these twice
                        tokens = logits.argmax(dim=1)

                    for i in range(len(actions)):
                        if eos[i]:
                            continue
                        actions[i].append(tokens[i].item())
                        eos[i] = tokens[i] == policy.eos_token
                    output_idx += 1

                # compute next states and rewards

                tactic_strings = policy.tokenizer.batch_decode([i[:-1] for i in actions])

                j = 0
                for i, __ in enumerate(start_states):

                    if done[i]:
                        continue

                    handler.send_tactic(tactic_strings[j], envs[i].proof_state)
                    response, _ = handler.receive_json()
                    has_error = "message" in response and response["message"].startswith("Lean error")
                    has_error = has_error or "messages" in response and any(m.severity == "error" for m in response["messages"])

                    if has_error:
                        log_rewards[i] = log(0.01)  # proof complete
                        done[i] = True
                        state_trajectories[i].append([policy.invalid_proof_token])
                        j += 1
                        continue
                    assert isinstance(response, LeanREPLNextProofState)
                    goals = response.goals
                    action_trajectories[i] = actions[j]
                    proof_state_history[i] = policy.tokenizer.decode([policy.goals_sep_id]).join(goals)

                    if not goals:
                        log_rewards[i] = log(10)  # proof complete
                        done[i] = True
                        state_trajectories[i].append([policy.successful_proof_token])
                    j += 1

                idx += 1

                for t in zip(state_trajectories, action_trajectories):

                    if not done[i]:
                        t[0].append([policy.incomplete_proof_token])

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

        # 2. call the model on each trajectory

        starting_states = [states[0] for states, __ in trajs]
        prompts = starting_states
        padded = policy.tokenizer.pad({"input_ids": prompts}, padding_side="left", return_tensors="pt")
        log_z_inputs = padded.input_ids.to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            log_z = policy.model.log_z(log_z_inputs)  # some duplicate computation happening here

        traj_lens = torch.tensor([len(actions) for __, actions in trajs], device=device)

        # stack each trajectory so we can use torch_scatter
        log_p_f = torch.zeros(traj_lens.sum(), device=device)
        log_p_b = torch.zeros(traj_lens.sum(), device=device)

        idx = 0
        # for each action, compute the sum of token log probs with prev state (p_f) and next state (p_b)
        for states, actions in trajs:
            for prev_state, action, next_state in zip(states[:-1], actions, states[:-1]):

                fwd_input = prev_state
                bck_input = next_state

                for t in action:

                    with torch.autocast(device_type=device, dtype=torch.float16):
                        # these are the forward and backward probability estimates for each token
                        log_p_f[idx] += policy.softmax(policy.model(torch.tensor([fwd_input]).to(device))[0, -1] / TEMPERATURE)[t].log()
                        log_p_b[idx] += policy.softmax(policy.model(torch.tensor([bck_input]).to(device))[0, -1] / TEMPERATURE)[t].log()

                    fwd_input += t
                    bck_input += t

                idx += 1

        # 3. compute TB loss with TLM backward policy

        batch_idx = torch.arange(len(traj_lens), device=device).repeat_interleave(traj_lens)

        traj_log_p_F = scatter(log_p_f, batch_idx, dim=0, dim_size=len(traj_lens), reduce="sum")
        traj_log_p_B = scatter(log_p_b, batch_idx, dim=0, dim_size=len(traj_lens), reduce="sum")

        back_loss = traj_log_p_B.mean()
        traj_log_p_B = traj_log_p_B.detach()
        log_rewards_tensor = torch.tensor(log_rewards, device=device)
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
            print(tb_loss_agg, back_loss_agg)

def get_precomputed_trajectories(start_theorems: List[Theorem], tokenizer: PreTrainedTokenizerFast) -> List[Tuple[List[List[int]], List[List[int]]]]:
    precomputed_trajectories = []
    for thm in tqdm(start_theorems):
        states = []
        actions = []
        for tactic in thm.traced_tactics:
            new_state = tokenizer.encode(tactic.state_before)
            # TODO: substitute this with policy.build_prompt()
            states.append(new_state) if not states else states.append([tokenizer.added_tokens_encoder["[STATESEP]"] ]+ new_state)
            actions.append(tokenizer.encode(tactic.tactic) + [tokenizer.added_tokens_encoder["[EOS]"]])
            trajectory = (states.copy(), actions.copy())
            precomputed_trajectories.append(trajectory)
    return precomputed_trajectories


def main():

    torch.set_float32_matmul_precision('high')

    parser = ArgumentParser()
    parser.add_argument("--n-layers", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=960)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    args = parser.parse_args()


    handler = LeanREPLHandler(Path("../leanproject"))
    start_theorems = list(get_start_theorems(LEAN_DOJO_PATH / "train.json"))

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="lean_tokenizer.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_layers = args.n_layers
    d_model = args.d_model
    config = MambaConfig(vocab_size=tokenizer.vocab_size, n_layer=n_layers, d_model=d_model)
    model = MambaLMHeadModelWrapper(config)
    model = torch.compile(model)
    torch.backends.cudnn.benchmark = True
    model.train()
    model.to(device)

    eos_id = tokenizer.added_tokens_encoder["[EOS]"]
    proofstate_id = tokenizer.added_tokens_encoder["[PROOFSTATE]"]
    proofstep_id = tokenizer.added_tokens_encoder["[PROOFSTEP]"]
    tactics_id = tokenizer.added_tokens_encoder["[TACTICS]"]
    tactics_sep_id = tokenizer.added_tokens_encoder["[SEP]"]
    proofstate_sep_id = tokenizer.added_tokens_encoder["[STATESEP]"]  # the policy sees a the list of proofstates we have transitioned to, separated by this token
    goals_sep_id = tokenizer.added_tokens_encoder["[GOALSEP]"]  # the current proof states is a list of goals separated by this token (maybe not necessary)

    # we need to be able to transition to unique leaf states, so end trajectories with the following tokens
    successful_proof_token = tokenizer.added_tokens_encoder["[SUC]"]
    incomplete_proof_token = tokenizer.added_tokens_encoder["[INC]"]
    invalid_proof_token = tokenizer.added_tokens_encoder["[INV]"]

    tokenizer.pad_token = "[PAD]"

    # TODO: we should pretrain with the same state formulation as the GFlowNet implementation
    policy = MambaPolicy.from_file("../model_small.pt", config, eos_id, proofstep_id, proofstate_id, tactics_id, tactics_sep_id, proofstate_sep_id,
                                   goals_sep_id, successful_proof_token, incomplete_proof_token, invalid_proof_token, tokenizer, device)

    gradient_accumulation_steps = args.gradient_accumulation_steps
    batch_size = args.batch_size
    rounds = args.rounds

    optimizer = optim.AdamW(model.get_non_z_params())
    z_optimizer = optim.AdamW(model.get_z_params())

    # TODO: add lr scheduler?
    precomputed_trajectories = get_precomputed_trajectories(start_theorems, tokenizer)

    train_gflownet(policy, start_theorems, precomputed_trajectories, handler, Path("../mathlib4"), optimizer, z_optimizer, gradient_accumulation_steps, batch_size, 0, rounds, device)

if __name__ == '__main__':
    main()
