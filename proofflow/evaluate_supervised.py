from collections import defaultdict

from transformers import PreTrainedTokenizerFast
import torch
from tqdm import tqdm
from proofflow.policy import MambaPolicy, ReProverPolicy
from proofflow.data import TrainSampleDataset
from proofflow.train import collate_train_samples
from pathlib import Path
import wandb
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import random
import numpy as np
import time

def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint to evaluate")
    parser.add_argument("json_path", type=str, help="Path to json file to evaluate")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--reprover", action="store_true", default=False, help="Use ReProver instead of Mamba.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    checkpoint_path = Path(args.checkpoint_path)
    json_path = Path(args.json_path)
    batch_size = args.batch_size
    num_workers = args.num_workers
    reprover = args.reprover

    assert checkpoint_path.exists() and json_path.exists()
    config = {"num_workers": num_workers, "batch_size": batch_size,
              "checkpoint_path": checkpoint_path, "json_path": json_path,
              "reprover": reprover, "seed": seed}

    wandb.init(project="proofflow", config=config)

    tokenizer = PreTrainedTokenizerFast.from_pretrained("./lean_tokenizer")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer.pad_token = "[PAD]"

    ds = TrainSampleDataset(json_path, filter_lake=False)
    if reprover:
        policy = ReProverPolicy.from_pretrained("cuda", False)
    else:
        policy = MambaPolicy.from_file(checkpoint_path, True, tokenizer, device)

    policy.model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_train_samples)
    torch.backends.cudnn.benchmark = True
    aggregated_metrics = defaultdict(list)
    start_time = time.perf_counter()
    for _ in range(5):
        for batch in tqdm(loader):
            metrics = policy.evaluate_batch(batch)
            for key in metrics:
                aggregated_metrics[key].append(metrics[key])
    end_time = time.perf_counter()
    # Compute mean
    results = {}
    for key in aggregated_metrics:
        aggregated_metric = aggregated_metrics[key]
        value = 0
        for metric in aggregated_metric:
            value += metric
        value /= len(aggregated_metric)
        results[key] = value

    results["time"] = end_time - start_time
    print("Evaluation results", results)
    wandb.log(results)
    wandb.finish(exit_code=0)

if __name__ == "__main__":
    main()