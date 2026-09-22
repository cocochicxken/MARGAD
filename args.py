"""Define MARGAD command-line options without import-time side effects."""

from __future__ import annotations

import argparse

from dataset_config import SUPPORTED_DATASETS, apply_dataset_defaults


ALPHA_ABLATION_MODES = (
    "full",
    "one_hop",
    "anonymous_two_hop",
    "fixed_equal_multiscale",
    "learned_no_coefficient_normalization",
    "learned_no_target_anonymization",
    "learned_no_degree_or_path_volume_normalization",
    "paper_matched_volume",
)
GAMMA_ABLATION_MODES = (
    "full",
    "fixed_shared_wavelet",
    "learned_global_shared_only",
    "learned_channel_specific_only",
    "dual_independent",
    "uncentered_cross_response",
)

# Historical CLI names retained for reproducibility:
# alpha = TA-DiffRef, beta = Global Deviation, gamma = WaveShift.


def build_parser() -> argparse.ArgumentParser:
    """Build the training and mechanism-ablation argument parser."""

    parser = argparse.ArgumentParser(
        description="Train the seven-dataset multi-filter graph anomaly detector."
    )
    parser.add_argument(
        "--dataset",
        default="Amazon",
        help="dataset name (case-insensitive): " + ", ".join(SUPPORTED_DATASETS),
    )
    parser.add_argument("--data_dir", default="dataset")

    # None means: use the selected dataset's documented README default.
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)

    parser.add_argument("--tests", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="Add this offset to each local run index (default preserves 0..runs-1).",
    )
    parser.add_argument(
        "--disable_monitor_auc",
        action="store_true",
        help=(
            "Do not compute label-based AUC during training epochs. "
            "Final AUC/AUPRC are still reported after training."
        ),
    )

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--batch_fanout", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--dgl_graph_on_gpu", type=int, choices=(0, 1))
    parser.add_argument(
        "--alpha_mode",
        choices=ALPHA_ABLATION_MODES,
        default="full",
        help="Optional alpha-mechanism ablation; full exactly preserves the default.",
    )
    parser.add_argument(
        "--gamma_mode",
        choices=GAMMA_ABLATION_MODES,
        default="full",
        help="Optional gamma-mechanism ablation; full exactly preserves the default.",
    )
    parser.add_argument(
        "--gamma_centering",
        type=int,
        choices=(0, 1),
        default=None,
        help=(
            "Center the shared and channel-specific Gamma responses before "
            "their discrepancy is computed (0/1). Default: 0 for Facebook, "
            "Elliptic and T-Social, 1 otherwise."
        ),
    )
    parser.add_argument(
        "--result_json",
        help="Optional machine-readable final metrics path (used by ablation_runner).",
    )
    parser.add_argument(
        "--diagnostics_json",
        help="Optional compact alpha/gamma diagnostics path after each seed.",
    )
    return parser


def parameter_parser(argv: list[str] | None = None):
    """Parse arguments, apply dataset fallbacks, and validate basic ranges."""

    options = build_parser().parse_args(argv)
    options, _ = apply_dataset_defaults(options)
    if options.epoch <= 0 or options.runs <= 0 or options.tests <= 0:
        raise ValueError("--epoch, --runs, and --tests must be positive.")
    if options.hidden_dim <= 0:
        raise ValueError("--hidden_dim must be positive.")
    if options.seed_offset < 0:
        raise ValueError("--seed_offset must be non-negative.")
    if options.gamma_centering is None:
        options.gamma_centering = options.dataset.strip().lower() not in (
            "facebook", "elliptic", "tsocial",
        )
    else:
        options.gamma_centering = bool(options.gamma_centering)
    return options
