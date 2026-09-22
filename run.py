"""Thin CLI entry point for training and evaluating MARGAD."""

from __future__ import annotations

from args import parameter_parser
from ablation_diagnostics import write_json_atomic
from Dataloader import load_full_graph, load_large_graph
from dataset_config import resolve_dataset
from full_graph import run_full_graph
from large_graph import run_large_graph


def run_experiment(options):
    """Dispatch one configured experiment to the full or sampled trainer."""

    spec = resolve_dataset(options.dataset)
    if spec.trainer == "full_graph":
        data = load_full_graph(spec, options.data_dir)
        return run_full_graph(options, spec, data)
    data = load_large_graph(spec, options.data_dir)
    return run_large_graph(options, spec, data)


def main(argv: list[str] | None = None):
    """Run the CLI workflow and optionally export machine-readable metrics."""

    options = parameter_parser(argv)
    print(options, flush=True)
    results = run_experiment(options)
    if options.result_json:
        write_json_atomic(options.result_json, {
            "schema_version": 1,
            "dataset": options.dataset,
            "alpha_mode": options.alpha_mode,
            "gamma_mode": options.gamma_mode,
            "seed_offset": options.seed_offset,
            "options": vars(options),
            "runs": [
                {
                    "auc": result.auc,
                    "auprc": result.auprc,
                    "best_monitor_epoch": result.best_monitor_epoch,
                    "best_monitor_auc": result.best_monitor_auc,
                    "best_selection_epoch": result.best_selection_epoch,
                    "best_selection_loss": result.best_selection_loss,
                    "checkpoint": str(result.checkpoint),
                    "train_seconds": result.train_seconds,
                    "inference_seconds": result.inference_seconds,
                    "total_seconds": result.total_seconds,
                    "peak_gpu_allocated_mib": result.peak_gpu_allocated_mib,
                    "peak_gpu_reserved_mib": result.peak_gpu_reserved_mib,
                }
                for result in results
            ],
        })
    return results


if __name__ == "__main__":
    main()
