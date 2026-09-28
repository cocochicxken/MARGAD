"""Export plot-ready T-Social scores for the fixed uncentered-Gamma model."""

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
from model import AdaptiveWaveletAffinity, GAD
from training_common import evaluate_numpy, minmax_numpy, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--output", default="tsocial_node_scores_uncentered.npz")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.85)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--alpha_mode", choices=AdaptiveWaveletAffinity.VALID_MODES, required=True)
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


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
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
        alpha_mode=args.alpha_mode,
        gamma_mode="full",
        gamma_centering=False,
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
        alpha_mode=args.alpha_mode,
        gamma_mode="full",
    ).to(trainer.device)
    model.load_state_dict(load_checkpoint(checkpoint))
    model.eval()

    with torch.no_grad():
        centers = trainer._estimate_centers(model)
        all_scores = trainer._collect_scores(model, centers).detach().cpu().numpy()

    labels = trainer.labels.detach().cpu().numpy().astype(np.int64, copy=False)
    if trainer.evaluation_index is None:
        node_ids = np.arange(labels.size, dtype=np.int64)
    else:
        node_ids = trainer.evaluation_index.detach().cpu().numpy().astype(np.int64)
    selected_labels = labels[node_ids]
    raw_scores = np.asarray(all_scores[node_ids], dtype=np.float64)
    auroc, auprc = evaluate_numpy(selected_labels, raw_scores)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "tsocial",
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "alpha_mode": args.alpha_mode,
        "gamma_mode": "full",
        "gamma_centering": False,
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
    print(
        f"T-Social scores exported: {output} | nodes={node_ids.size} "
        f"| AUROC={auroc * 100:.4f} | AUPRC={auprc * 100:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
