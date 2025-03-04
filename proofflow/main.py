from lean_repl_py import LeanREPLHandler
from proofflow.policy import Policy
from torch import nn
from transformers import AutoTokenizer


class Model(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 30)
        self.lstm1 = nn.LSTM(30, 30, 1)
        self.linear1 = nn.Linear(30, 50)
        self.lstm2 = nn.LSTM(50, 50, 1)
        self.linear2 = nn.Linear(50, vocab_size)

    def forward(self, x):
        assert len(x.shape) == 1  # [seq_len], no batch dim
        x = self.emb(x)
        x, _ = self.lstm1(x)
        x = self.linear1(x)
        x, _ = self.lstm2(x)
        return self.linear2(x)


handler = LeanREPLHandler()

handler.send_command("""theorem p_and_q (p q : Prop) (a : p) (b : q): p ∧ q := by
    sorry""")

response, _ = handler.receive_json()

print(response)
proof_state = response["sorries"][0]

handler.send_tactic("constructor", proof_state.proof_state)

response, _ = handler.receive_json()

tokenizer = AutoTokenizer.from_pretrained("kaiyuy/leandojo-lean4-tacgen-byt5-small")  # reprover

# Vocab size is 256, but we use the extra special tokens at 260, so I increase this to 263
model = Model(263)
eos_token = tokenizer.eos_token_id
# The ids starting at 259 are unused in regular t5, so I simply use them for our special cases
proof_step_id = 259
goal_id = 260
tactics_id = 261
tactics_sep_id = 262

policy = Policy(model, eos_token, proof_step_id, goal_id, tactics_id, tactics_sep_id, tokenizer)

print(policy.next_tactic(response.goals[0]))
