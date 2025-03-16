import tempfile

from transformers import PreTrainedTokenizerFast
import torch
from proofflow.policy import MambaPolicy
from proofflow.data import ProofStateDataset
from proofflow.train_gfn import MCTS, Node, sample_mcts_trajectories, _compute_log_rewards
from pathlib import Path
from lean_repl_py import LeanREPLHandler, LeanREPLNextProofState
from tempfile import TemporaryDirectory
from typing import List, Optional

def _process_single_env(handler: LeanREPLHandler, tactics: List[str], proof_state_indices: List[int],
                              proven: List[bool], invalid: List[bool], indices: List[Optional[int]],
                              goals: List[Optional[List[str]]], times: List[float]):
    assert len(proven) == len(invalid) == len(indices) == len(goals) == 0
    for tactic, proof_state_idx in zip(tactics, proof_state_indices, strict=True):

        handler.send_tactic(tactic, proof_state_idx)
        response, _ = handler.receive_json()
        has_error = "message" in response and response["message"].startswith("Lean error")
        has_error = has_error or "messages" in response and any(msg.severity == "error" for msg in response["messages"])
        if has_error:
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


def _process_all_envs(handlers: List[LeanREPLHandler], tactics: List[List[str]],
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
    return proven, invalid, indices, goals, times


def _envs_expand(
        handlers: list[LeanREPLHandler],
        tactics: list[list[str]],
        proof_state_indices: list[list[int]]
):
    """Sends one tactic at a time per handler"""
    assert len(handlers) == len(tactics) == len(proof_state_indices)
    proven, invalid, indices, goals, times = _process_all_envs(handlers, tactics, proof_state_indices)
    return proven, invalid, indices, goals, times


tokenizer = PreTrainedTokenizerFast.from_pretrained("./lean_tokenizer")

device = "cuda" if torch.cuda.is_available() else "cpu"

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
with TemporaryDirectory() as tmpdir:
    handler_fac = lambda: LeanREPLHandler(Path("./leanproject"))

    ds = ProofStateDataset(Path("proof_flow_theorems.json"), handler_fac, Path("./mathlib4"), Path(tmpdir))
    policy = MambaPolicy.from_file(Path("./modelnewesttokenizer.pt"), True, tokenizer, device)
    policy.model.eval()
    # policy.model = torch.compile(policy.model)
    torch.backends.cudnn.benchmark = True
    batch = [ds[i] for i in range(len(ds))]
    thms, paths = [elem[0] for elem in batch], [elem[1] for elem in batch]
    handlers = [handler_fac() for _ in paths]
    # Start all REPLs
    proof_states = [handler.unpickle_proof_state(path) for handler, path in zip(handlers, paths)]
    # Gather
    proof_states = [proof_state for proof_state, env in proof_states]

    # Unlink them all, not needed anymore
    assert all(path.exists() for path in paths)
    for path in paths:
        path.unlink(missing_ok=True)  # missing ok because of repeats
    assert all(len(proof_state.goals) == 1 for proof_state in proof_states)
    nodes = [Node(proof_state.goals[0], proof_state_idx=proof_state.proof_state,
                  metadata={"theoremname": thm.full_name, "theoremfile": thm.file_path}) for proof_state, thm in
             zip(proof_states, thms)]
    mcts = [MCTS(node) for node in nodes]


    while not all(node.done for node in mcts):        
        for _ in range(5000):
            if all(node.done for node in mcts):
                print("Breaking!")
                break
            currents = []
            for node in mcts:
                if node.done:
                    continue
                current = node.select()
                assert not current.has_been_expanded
                assert not current.branch_is_done
                currents.append(current)

            end_states = [current.proof_state for current in currents]
            histories = [current.previous_states for current in currents]

            tactic_strings = []
            for end_state, history in zip(end_states, histories):
                tactic_strings.append(policy.next_tactics(end_state, 100, None, history, temperature=1))

            current_handlers = [handlers[node_idx] for node_idx in range(len(mcts)) if
                                not mcts[node_idx].done]
            proven, invalid, indices, goals, times_current = _envs_expand(current_handlers, tactic_strings,
                                                                          [[current.proof_state_idx] * len(
                                                                              tactic_strings[current_idx]) for
                                                                           current_idx, current in
                                                                           enumerate(currents)], )
            current_idx = 0

            for i, node in enumerate(mcts):

                if node.done:
                    continue

                rewards = _compute_log_rewards(proven[current_idx], invalid[current_idx], times_current[current_idx], len(currents[current_idx].previous_states) + 1)
                # Only passes on valid tactics to expand, we might want to change this

                tactics = [t for t, p in zip(tactic_strings[current_idx], invalid) if not p]
                goals_node = [g for g, p in zip(goals[current_idx], invalid) if not p]
                times_current_node = [t for t, p in zip(times_current[current_idx], invalid) if not p]
                indices_node = [index for index, p in zip(indices[current_idx], invalid) if not p]
                rewards = [r for r, p in zip(rewards, invalid) if not p]

                currents[current_idx].expand(tactics, goals_node, times_current_node, rewards, indices_node)
                if any(proven[current_idx]):
                    print("Proof: ", proven[current_idx])
                    node.done = True
                    node.solved = True
                    node.proof += tactic_strings[current_idx][proven[current_idx].index(True)]
                    node.last_tactic = tactic_strings[current_idx][proven[current_idx].index(True)]
                # Edge case, if we only have invalid tactics, there is no way to continue
                elif node.root.branch_is_done:
                    node.done = True
                    node.solved = False
                current_idx += 1

if all(any(proven[current_idx]) for current_idx in range(len(proof_states))):
    print("All proven!")
for node in mcts:
    print(node.solved)
    print(node.proof)