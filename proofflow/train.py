from collections import defaultdict

from torch import nn
import torch
from proofflow.model.ffm import FFM
from proofflow.data import parse_json, LEAN_DOJO_PATH, TheoremDataset, TrainSampleDataset, TrainingSample
from pathlib import Path
from torch.utils.data import DataLoader
from proofflow.policy import Policy
from transformers import PreTrainedTokenizerFast
import torch.optim as optim
import wandb
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm import MambaLMHeadModel
from tqdm import tqdm


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
        #self.block1 = ModelBlock(20, 20, 64, 3, 3, 0.1)
        #self.block2 = ModelBlock(20, 20, 64, 3, 3, 0.1)
        self.block3 = ModelBlock(20, vocab_size, 64, 3, 3, 0.1)

    def forward(self, x):
        if x.shape == 1:
            x = x[None]
        assert len(x.shape) == 2  # [batch_dim, seq_len]
        x = self.emb(x)  # [batch_dim, seq_len, emb_dim]
        x = self.norm(x)
        x = self.dropout(x)
        #x = self.block1(x)
        #x = self.block2(x)
        return self.block3(x)


class MambaLMHeadModelWrapper(MambaLMHeadModel):
   def forward(self, x):
       return super().forward(x).logits


def collate_train_samples(batch: list[TrainingSample]):
    return batch


def evaluate(policy: Policy, data: TrainSampleDataset, eval_batch_size: int = 64) -> dict[str, float]:
    metrics = defaultdict(float)
    print("Evaluating")
    for batch in tqdm(DataLoader(data, batch_size=eval_batch_size, shuffle=True, collate_fn=collate_train_samples)):
        metrics_batch = policy.evaluate_batch(batch)
        for key, value in metrics_batch.items():
            metrics[key] += value
    for key in metrics:
        metrics[key] /= len(data) / eval_batch_size
    return metrics


def train_loop(policy: Policy, data: TrainSampleDataset, optimizer: optim.Optimizer, gradient_accumulation_steps: int,
               batch_size: int, eval_steps: int, valid_data: TrainSampleDataset, device: str):
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_train_samples)
    policy.model.train()
    optimizer.zero_grad()
    current_step = 0
    for epoch in range(3):
        for batch in tqdm(data_loader):
            loss = policy.train_batch(batch) / gradient_accumulation_steps
            wandb.log({"loss": loss})
            loss.backward()
            if (current_step + 1) % gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if current_step % eval_steps == 0:
                metrics = evaluate(policy, valid_data, 3)
                wandb.log(metrics)
                print(metrics)
            current_step += 1
        print(f"Epoch {epoch} done")
        print("Saving model")
        policy.save(Path("model.pt"))


def main():
    train_data = TrainSampleDataset(LEAN_DOJO_PATH / "train.json")
    valid_data = TrainSampleDataset(LEAN_DOJO_PATH / "val.json")
    eval_data = TheoremDataset(LEAN_DOJO_PATH / "val.json")
    test_data = TheoremDataset(LEAN_DOJO_PATH / "test.json")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="lean_tokenizer.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = Model(tokenizer.vocab_size).to(device)

    config = MambaConfig(vocab_size=tokenizer.vocab_size, n_layer=30, d_model=960)
    model = MambaLMHeadModelWrapper(config).to(device)

    eos_id = tokenizer.added_tokens_encoder["[EOS]"]
    proofstate_id = tokenizer.added_tokens_encoder["[PROOFSTATE]"]
    proofstep_id = tokenizer.added_tokens_encoder["[PROOFSTEP]"]
    tactics_id = tokenizer.added_tokens_encoder["[TACTICS]"]
    tactics_sep_id = tokenizer.added_tokens_encoder["[SEP]"]
    tokenizer.pad_token = "[PAD]"
    policy = Policy(model, eos_id, proofstep_id, proofstate_id, tactics_id, tactics_sep_id, tokenizer, device)

    gradient_accumulation_steps = 10
    batch_size = 2
    eval_steps = 10_000
    optimizer = optim.AdamW(model.parameters())
    wandb.init(project="proofflow")
    train_loop(policy, train_data, optimizer, gradient_accumulation_steps, batch_size, eval_steps, valid_data, device)


if __name__ == '__main__':
    main()
