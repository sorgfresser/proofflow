from lean_repl_py import LeanREPLHandler
from proofflow.policy import MambaPolicy
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from mamba_ssm.models.config_mamba import MambaConfig
import torch

handler = LeanREPLHandler()

handler.send_command("""theorem p_and_q (p q : Prop) (a : p) (b : q): p ∧ q := by
    sorry""")

response, _ = handler.receive_json()

print(response)
proof_state = response["sorries"][0]

handler.send_tactic("constructor", proof_state.proof_state)

response, _ = handler.receive_json()

tokenizer = PreTrainedTokenizerFast.from_pretrained("./lean_tokenizer")
eos_id = tokenizer.added_tokens_encoder["[EOS]"]
proofstate_id = tokenizer.added_tokens_encoder["[PROOFSTATE]"]
proofstep_id = tokenizer.added_tokens_encoder["[PROOFSTEP]"]
tactics_id = tokenizer.added_tokens_encoder["[TACTICS]"]
tactics_sep_id = tokenizer.added_tokens_encoder["[SEP]"]
tokenizer.pad_token = "[PAD]"
proofstate_sep_id = tokenizer.added_tokens_encoder[
    "[STATESEP]"]  # the policy sees a the list of proofstates we have transitioned to, separated by this token

# we need to be able to transition to unique leaf states, so end trajectories with the following tokens
successful_proof_token = tokenizer.added_tokens_encoder["[SUC]"]
incomplete_proof_token = tokenizer.added_tokens_encoder["[INC]"]
invalid_proof_token = tokenizer.added_tokens_encoder["[INV]"]

eos_token = tokenizer.eos_token_id
config = MambaConfig(vocab_size=tokenizer.vocab_size, n_layer=12, d_model=240)

device = "cuda" if torch.cuda.is_available() else "cpu"

policy = MambaPolicy.from_file("../model_small.pt", False, tokenizer, device)

print(policy.next_tactics(response.goals[0], k=10))
print(policy.next_tactics(response.goals, k=10, temperature=0.1))
print(policy.next_tactic(response.goals[0], temperature=0.1))
print(policy.next_tactic(response.goals, temperature=0.1))
