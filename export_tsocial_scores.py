"""Export plot-ready MARGAD scores for a specified T-Social checkpoint.

Exactly one checkpoint source is required:
- --checkpoint PATH : explicit checkpoint file, e.g. an existing centered final model;
- --checkpoint_run N : best_model_run{N}.pth inside the process working directory
  (the per-dataset run directory written by final_10run_efficiency.py).

--gamma_centering must match the training state of the checkpoint.  The exporter
verifies this: a centered full-mode model requires two population centres
(gamma_thick / gamma_thin) while an uncentered model requires none, because the
discrepancy is then computed without subtracting them.  The verification result
is recorded in the exported metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from Dataloader import load_large_graph
from dataset_config import resolve_dataset
from large_graph import TSocialTrainer
from model import GAD
from training_common import evaluate_numpy, minmax_numpy, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="Explicit checkpoint file path.")
    source.add_argument(
        "--checkpoint_run",
        type=int,
        help="Run index N: loads best_model_run{N}.pth from the working directory.",
    )
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.85)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--gamma_centering", type=int, choices=(0, 1), default=1)
    parser.add_argument("--batch_fanout", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=51200)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dgl_graph_on_gpu", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_checkpoint_source(args: argparse.Namespace) -> tuple[Path, str | None]:
    if args.checkpoint_run is not None:
        path = Path.cwd() / f"best_model_run{args.checkpoint_run}.pth"
        return path, f"run{args.checkpoint_run}"
    return Path(args.checkpoint).resolve(), None


def default_output(gamma_centering: bool) -> str:
    state = "centered" if gamma_centering else "uncentered"
    return f"tsocial_node_scores_{state}.npz"


def main() -> None:
    """Validate the checkpoint policy, run inference, and export scores."""

    args = parse_args()
    checkpoint, checkpoint_run = resolve_checkpoint_source(args)
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    set_seed(args.seed)
    spec = resolve_dataset("tsocial")
    options = SimpleNamespace(
        device=args.device,
        hidden_dim=args.hidden_dim,
        dropout=0.0,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        alpha_mode="full",
        gamma_mode="full",
        gamma_centering=bool(args.gamma_centering),
        batch_fanout=args.batch_fanout,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        dgl_graph_on_gpu=args.dgl_graph_on_gpu,
    )
    trainer = TSocialTrainer(options, spec, load_large_graph(spec, Path(args.data_dir)))
    model = GAD(
        feat_size=trainer.feature_store.size(1),
        hidden_size=args.hidden_dim,
        dropout=0.0,
        alpha_mode="full",
        gamma_mode="full",
    ).to(trainer.device)
    model.load_state_dict(load_checkpoint(checkpoint))
    model.eval()

    with torch.no_grad():
        centers = trainer._estimate_centers(model)
        gamma_keys = {"gamma_thick", "gamma_thin"}
        if args.gamma_centering:
            if not gamma_keys.issubset(centers):
                raise RuntimeError(
                    "Centering mismatch: --gamma_centering 1 requested, but the "
                    "checkpoint's full-mode model estimated no Gamma population "
                    f"centres (got {sorted(centers)}). Use --gamma_centering 0 "
                    "for this checkpoint."
                )
        elif gamma_keys & centers:
            raise RuntimeError(
                "Centering mismatch: --gamma_centering 0 requested, but the "
                "checkpoint's full-mode model estimated Gamma population centres "
                f"({sorted(centers)}). Use --gamma_centering 1 for this checkpoint."
            )
        all_scores = trainer._collect_scores(model, centers).detach().cpu().numpy()

    labels = trainer.labels.detach().cpu().numpy().astype(np.int64, copy=False)
    if trainer.evaluation_index is None:
        node_ids = np.arange(labels.size, dtype=np.int64)
    else:
        node_ids = trainer.evaluation_index.detach().cpu().numpy().astype(np.int64)
    selected_labels = labels[node_ids]
    raw_scores = np.asarray(all_scores[node_ids], dtype=np.float64)
    auroc, auprc = evaluate_numpy(selected_labels, raw_scores)

    output = Path(args.output or default_output(bool(args.gamma_centering))).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "tsocial",
        "checkpoint": str(checkpoint),
        "checkpoint_run": checkpoint_run,
        "seed": args.seed,
        "gamma_mode": "full",
        "gamma_centering": bool(args.gamma_centering),
        "centering_consistency": "verified",
        "estimated_center_keys": sorted(centers),
        "score_key": "scores",
        "score_range": "[0, 1] min-max normalized over exported nodes",
        "recomputed_auroc": float(auroc),
        "recomputed_auprc": float(auprc),
    }
    np.savez_compressed(
        output,
        node_ids=node_ids,
        labels=selected_labels,
        scores=minmax_numpy(raw_scores).astype(np.float32, copy=False),
        fused_scores_raw=raw_scores.astype(np.float32, copy=False),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    state = "centered" if args.gamma_centering else "uncentered"
    print(
        f"T-Social {state} scores exported: {output} | nodes={node_ids.size} "
        f"| AUROC={auroc * 100:.4f} | AUPRC={auprc * 100:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
