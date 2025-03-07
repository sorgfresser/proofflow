from torch import nn
import torch
from proofflow.model.ffm import FFM
from proofflow.data import parse_json, LEAN_DOJO_PATH, TheoremDataset, TrainSampleDataset, TrainingSample
from pathlib import Path
from torch.utils.data import DataLoader
from proofflow.policy import Policy
from transformers import PreTrainedTokenizerFast
import torch.optim as optim


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
        self.emb = nn.Embedding(vocab_size, 30)
        self.norm = nn.LayerNorm(30)
        self.dropout = nn.Dropout(p=0.1)
        self.block1 = ModelBlock(30, 50, 64, 12, 12, 0.1)
        self.block2 = ModelBlock(50, 100, 64, 12, 12, 0.1)
        self.block3 = ModelBlock(100, vocab_size, 64, 12, 12, 0.1)

    def forward(self, x):
        if x.shape == 1:
            x = x[None]
        assert len(x.shape) == 2  # [batch_dim, seq_len]
        x = self.emb(x)  # [batch_dim, seq_len, emb_dim]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.block3(x)

def collate_train_samples(batch: list[TrainingSample]):
    return batch


train_data = TrainSampleDataset(LEAN_DOJO_PATH / "train.json")
valid_data = TheoremDataset(LEAN_DOJO_PATH / "val.json")
test_data = TheoremDataset(LEAN_DOJO_PATH / "test.json")

tokenizer = PreTrainedTokenizerFast(tokenizer_file="lean_tokenizer.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Model(tokenizer.vocab_size).to(device)
eos_id = tokenizer.added_tokens_encoder["[EOS]"]
proofstate_id = tokenizer.added_tokens_encoder["[PROOFSTATE]"]
proofstep_id = tokenizer.added_tokens_encoder["[PROOFSTEP]"]
tactics_id = tokenizer.added_tokens_encoder["[TACTICS]"]
tactics_sep_id = tokenizer.added_tokens_encoder["[SEP]"]
tokenizer.pad_token = "[PAD]"
policy = Policy(model, eos_id, proofstep_id, proofstate_id, tactics_id, tactics_sep_id, tokenizer)
# train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True)
# valid_data_loader = DataLoader(valid_data, batch_size=64, shuffle=True)


data_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_train_samples)
optimizer = optim.AdamW(model.parameters())

for epoch in range(3):
    for batch in data_loader:
        loss = policy.train_batch(batch)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
