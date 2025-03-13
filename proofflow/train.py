from collections import defaultdict

from torch import nn
import torch
from proofflow.model.ffm import FFM
from proofflow.data import parse_json, LEAN_DOJO_PATH, TheoremDataset, TrainSampleDataset, TrainingSample
from pathlib import Path
from torch.utils.data import DataLoader
from proofflow.policy import Policy, MambaLMHeadModelWrapper, MambaPolicy
from transformers import PreTrainedTokenizerFast
import torch.optim as optim
import wandb
from mamba_ssm.models.config_mamba import MambaConfig
from tqdm import tqdm
from argparse import ArgumentParser


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

    def forward(self, x):
        if x.shape == 1:
            x = x[None]
        assert len(x.shape) == 2  # [batch_dim, seq_len]
        x = self.emb(x)  # [batch_dim, seq_len, emb_dim]
        x = self.norm(x)
        x = self.dropout(x)
        # x = self.block1(x)
        # x = self.block2(x)
        return self.block3(x)


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


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, default="model.pt")
    parser.add_argument("--reload-checkpoint", action="store_true", default=False)
    parser.add_argument("--n-layers", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=10_000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    args = parser.parse_args()

    train_data = TrainSampleDataset(LEAN_DOJO_PATH / "train.json")
    valid_data = TrainSampleDataset(LEAN_DOJO_PATH / "val.json")
    eval_data = TheoremDataset(LEAN_DOJO_PATH / "val.json")
    test_data = TheoremDataset(LEAN_DOJO_PATH / "test.json")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file="lean_tokenizer.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = Model(tokenizer.vocab_size).to(device)
    n_layers = args.n_layers
    d_model = args.d_model
    config = MambaConfig(vocab_size=tokenizer.vocab_size, n_layer=n_layers, d_model=d_model)
    model = MambaLMHeadModelWrapper(config)
    if args.reload_checkpoint:
        model.load_state_dict(torch.load(args.checkpoint_path))
    model.train()
    model.to(device)

    eos_id = tokenizer.added_tokens_encoder["[EOS]"]
    proofstate_id = tokenizer.added_tokens_encoder["[PROOFSTATE]"]
    proofstep_id = tokenizer.added_tokens_encoder["[PROOFSTEP]"]
    tactics_id = tokenizer.added_tokens_encoder["[TACTICS]"]
    tactics_sep_id = tokenizer.added_tokens_encoder["[SEP]"]
    proofstate_sep_id = tokenizer.added_tokens_encoder["[STATESEP]"]
    goals_sep_id = tokenizer.added_tokens_encoder["[GOALSEP]"]
    successful_proof_token = tokenizer.added_tokens_encoder["[SUC]"]
    incomplete_proof_token = tokenizer.added_tokens_encoder["[INC]"]
    invalid_proof_token = tokenizer.added_tokens_encoder["[INV]"]
    tokenizer.pad_token = "[PAD]"
    policy = MambaPolicy(model, config, eos_id, proofstep_id, proofstate_id, tactics_id, tactics_sep_id, tokenizer, device)

    gradient_accumulation_steps = args.gradient_accumulation_steps
    batch_size = args.batch_size
    eval_steps = args.eval_steps
    epochs = args.epochs
    eval_batch_size = args.eval_batch_size
    optimizer = optim.AdamW(model.parameters())
    config = {"gradient_accumulation_steps": gradient_accumulation_steps, "batch_size": batch_size, "epochs": epochs,
              "eval_steps": eval_steps, "n_layers": n_layers, "d_model": d_model, "eval_batch_size": eval_batch_size}
    wandb.init(project="proofflow", config=config)
    train_loop(policy, train_data, optimizer, gradient_accumulation_steps, batch_size, eval_steps, valid_data,
               Path(args.checkpoint_path), eval_batch_size=eval_batch_size, epochs=epochs)
    wandb.finish(exit_code=0)


if __name__ == '__main__':
    main()
