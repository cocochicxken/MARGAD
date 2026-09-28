"""Single source of truth for the seven supported datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TrainerKind = Literal["full_graph", "large_graph"]
LoaderKind = Literal["mat", "elliptic_csv", "dgl_full", "dgl_large"]


@dataclass(frozen=True)
class TrainingDefaults:
    hidden_dim: int
    lr: float
    epoch: int
    alpha: float
    beta: float
    gamma: float
    batch_size: int = 51200
    eval_batch_size: int = 51200
    batch_fanout: int = 8
    num_workers: int = 0
    dgl_graph_on_gpu: int = 1


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    cli_name: str
    loader: LoaderKind
    trainer: TrainerKind
    defaults: TrainingDefaults
    normalize_alpha: bool = True
    standardize_features: bool = False
    gamma_normalize_bands: bool = True
    gamma_input_tanh: bool = True
    gamma_coefficient_tanh: bool = True


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "facebook", "Facebook", "mat", "full_graph",
        TrainingDefaults(64, 2e-3, 80, 1.0, 0.15, 0.60),
    ),
    DatasetSpec(
        "reddit", "Reddit", "mat", "full_graph",
        TrainingDefaults(128, 3e-3, 100, 1.0, 0.35, 0.30),
        standardize_features=True,
        # Experimental Reddit variant: preserve the normalized Gamma geometry
        # while avoiding tanh on the Gamma signal and learned coefficients.
        gamma_normalize_bands=True,
        gamma_input_tanh=False,
        gamma_coefficient_tanh=False,
    ),
    DatasetSpec(
        "amazon", "Amazon", "mat", "full_graph",
        TrainingDefaults(128, 2e-3, 100, 1.0, 0.40, 0.40),
    ),
    DatasetSpec(
        "elliptic", "elliptic", "elliptic_csv", "full_graph",
        TrainingDefaults(128, 1e-3, 70, 1.0, 0.15, 1.50),
        normalize_alpha=False,
    ),
    DatasetSpec(
        "yelpchi", "YelpChi", "mat", "full_graph",
        TrainingDefaults(64, 2e-3, 70, 1.0, 0.15, 0.90),
        # YelpChi follows the requested raw-Gamma band geometry.  Alpha and
        # Beta keep their existing normalization behavior.
        gamma_normalize_bands=False,
    ),
    DatasetSpec(
        "tfinance", "tfinance", "dgl_full", "full_graph",
        TrainingDefaults(64, 3e-3, 100, 1.0, 0.25, 0.90),
    ),
    DatasetSpec(
        "tsocial", "tsocial", "dgl_large", "large_graph",
        TrainingDefaults(64, 3e-3, 25, 1.0, 0.30, 0.30),
        # Preserve T-Social's historical Gamma geometry: the Gamma input is
        # already raw H and thick/thin bands remain unnormalized.
        gamma_normalize_bands=False,
    ),
)

_BY_KEY = {spec.key: spec for spec in DATASET_SPECS}
SUPPORTED_DATASETS = tuple(spec.cli_name for spec in DATASET_SPECS)


def resolve_dataset(name: str) -> DatasetSpec:
    """Resolve a dataset name case-insensitively and reject unsupported routes."""
    try:
        return _BY_KEY[name.lower()]
    except KeyError as error:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset {name!r}. Choose one of: {supported}.") from error


def apply_dataset_defaults(options):
    """Fill omitted CLI values from the selected dataset's README configuration."""
    spec = resolve_dataset(options.dataset)
    defaults = spec.defaults
    options.dataset = spec.cli_name
    for field in (
        "hidden_dim", "lr", "epoch", "alpha", "beta", "gamma",
        "batch_size", "eval_batch_size", "batch_fanout", "num_workers",
        "dgl_graph_on_gpu",
    ):
        if getattr(options, field) is None:
            setattr(options, field, getattr(defaults, field))
    return options, spec
