import tempfile

from transformers import PreTrainedTokenizerFast
import torch
from proofflow.policy import MambaPolicy, ReProverPolicy
from proofflow.data import ProofStateDataset
from proofflow.train_gfn import MCTS, Node, sample_mcts_trajectories, collate_skip_none, evaluate
from pathlib import Path
from lean_repl_py import LeanREPLHandler, LeanREPLNextProofState
from tempfile import TemporaryDirectory
from typing import List, Optional
import time
import wandb
from torch.utils.data import DataLoader
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint to evaluate")
    parser.add_argument("json_path", type=str, help="Path to json file to evaluate")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-tactics", type=int, default=32)
    parser.add_argument("--search-time", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--reprover", action="store_true", default=False)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    json_path = Path(args.json_path)
    search_time = args.search_time
    batch_size = args.batch_size
    num_tactics = args.num_tactics
    num_workers = args.num_workers
    reprover = args.reprover

    assert checkpoint_path.exists() and json_path.exists()
    config = {"num_workers": num_workers, "batch_size": batch_size, "num_tactics": num_tactics,
              "search_time": search_time, "checkpoint_path": checkpoint_path, "json_path": json_path,
              "reprover": reprover}
    wandb.init(project="proofflow", config=config)

    tokenizer = PreTrainedTokenizerFast.from_pretrained("./lean_tokenizer")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer.pad_token = "[PAD]"
    with TemporaryDirectory() as tmpdir:
        handler_fac = lambda: LeanREPLHandler(Path("./leanproject"))
        ds = ProofStateDataset(json_path, handler_fac, Path("./mathlib4"), Path(tmpdir), filter_lake=False)
        if reprover:
            policy = ReProverPolicy.from_pretrained("cuda", False)
        else:
            policy = MambaPolicy.from_file(checkpoint_path, True, tokenizer, device)
        policy.model.eval()
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_skip_none,
                            num_workers=num_workers)
        torch.backends.cudnn.benchmark = True

        results = evaluate(policy, loader, handler_fac, device, 1, search_time, num_tactics, batch_size, "")
        print("Evaluation results", results)
        wandb.log(results)
    wandb.finish(exit_code=0)

if __name__ == "__main__":
    main()