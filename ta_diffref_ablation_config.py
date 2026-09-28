"""Frozen protocol for the final-configuration TA-DiffRef A0--A6 study."""

from __future__ import annotations

from dataclasses import dataclass


EXPERIMENT_VERSION = "ta-diffref-a0-a6-final-config-v1"
DEFAULT_SEEDS = tuple(range(5))


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    cli_name: str
    hidden_dim: int
    learning_rate: float
    epochs: int
    global_deviation_weight: float
    waveshift_weight: float
    gamma_centering: int
    trainer: str


@dataclass(frozen=True)
class VariantConfig:
    code: str
    alpha_mode: str
    reference_operator: str
    scale: str
    per_order_calibration: str
    target_return_removed: str
    scale_weights: str


DATASETS = (
    DatasetConfig("facebook", "Facebook", 64, 0.003, 70, 0.15, 0.50, 1, "full_graph"),
    DatasetConfig("reddit", "Reddit", 128, 0.003, 110, 0.35, 1.45, 1, "full_graph"),
    DatasetConfig("yelpchi", "YelpChi", 64, 0.003, 65, 0.15, 1.05, 1, "full_graph"),
    DatasetConfig("tfinance", "tfinance", 64, 0.003, 85, 1.00, 0.05, 1, "full_graph"),
    DatasetConfig("elliptic", "elliptic", 64, 0.001, 70, 0.30, 1.00, 1, "full_graph"),
    DatasetConfig("tsocial", "tsocial", 64, 0.003, 10, 0.85, 0.75, 0, "large_graph"),
)


VARIANTS = (
    VariantConfig(
        "A0", "one_hop", "P", "one_order", "random_walk_row_normalization",
        "one_hop_has_no_self_loop", "not_applicable",
    ),
    VariantConfig(
        "A1", "symmetric_one_hop", "S", "one_order", "symmetric_degree_form",
        "one_hop_has_no_self_loop", "not_applicable",
    ),
    VariantConfig(
        "A2", "raw_equal_multiscale", "0.5*B + 0.5*B^2", "two_orders", "no",
        "no", "fixed_equal",
    ),
    VariantConfig(
        "A3", "fixed_equal_no_target_anonymization", "0.5*P + 0.5*P^2",
        "two_orders", "yes", "no", "fixed_equal",
    ),
    VariantConfig(
        "A4", "raw_equal_anonymous_no_renormalization",
        "0.5*B + 0.5*(B^2-diag(diag(B^2)))", "two_orders", "no", "yes",
        "fixed_equal",
    ),
    VariantConfig(
        "A5", "fixed_equal_multiscale", "0.5*P + 0.5*Gamma^(2)",
        "two_orders", "yes", "yes", "fixed_equal",
    ),
    VariantConfig(
        "A6", "full", "three-head adaptive P/Gamma^(2)", "two_orders", "yes",
        "yes", "three_head_adaptive",
    ),
)


PAIRINGS = (
    ("A2", "A3", "per-order calibration with target return retained"),
    ("A4", "A5", "per-order calibration after target-return removal"),
    ("A2", "A4", "target-return removal under raw propagation"),
    ("A3", "A5", "target anonymization under calibrated propagation"),
    ("A5", "A6", "fixed equal versus three-head adaptive scale integration"),
)


def _split_tokens(values) -> list[str]:
    tokens: list[str] = []
    for value in values or ():
        tokens.extend(part.strip() for part in str(value).split(",") if part.strip())
    return tokens


def select_datasets(values=None) -> tuple[DatasetConfig, ...]:
    tokens = _split_tokens(values)
    if not tokens or any(token.lower() == "all" for token in tokens):
        return DATASETS
    lookup = {}
    for spec in DATASETS:
        lookup[spec.key.lower()] = spec
        lookup[spec.cli_name.lower()] = spec
    unknown = [token for token in tokens if token.lower() not in lookup]
    if unknown:
        raise ValueError("Unknown dataset(s): " + ", ".join(unknown))
    selected = {lookup[token.lower()].key for token in tokens}
    return tuple(spec for spec in DATASETS if spec.key in selected)


def select_variants(values=None) -> tuple[VariantConfig, ...]:
    tokens = _split_tokens(values)
    if not tokens or any(token.lower() == "all" for token in tokens):
        return VARIANTS
    lookup = {variant.code.lower(): variant for variant in VARIANTS}
    unknown = [token for token in tokens if token.lower() not in lookup]
    if unknown:
        raise ValueError("Unknown variant(s): " + ", ".join(unknown))
    selected = {token.lower() for token in tokens}
    return tuple(variant for variant in VARIANTS if variant.code.lower() in selected)


def parse_seeds(values=None) -> tuple[int, ...]:
    tokens = _split_tokens(values)
    if not tokens:
        return DEFAULT_SEEDS
    seeds: list[int] = []
    for token in tokens:
        if "-" in token:
            first, last = token.split("-", 1)
            start, stop = int(first), int(last)
            if stop < start:
                raise ValueError(f"Invalid seed range: {token}")
            seeds.extend(range(start, stop + 1))
        else:
            seeds.append(int(token))
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative.")
    return tuple(dict.fromkeys(seeds))


def operator_protocol(dataset: DatasetConfig, variant: VariantConfig) -> dict[str, str]:
    if dataset.trainer == "full_graph":
        return {
            "execution": "exact sparse full-graph propagation",
            "degree_source": "degree of B after self-loop removal",
            "two_hop_definition": "recursive sparse multiplication; no dense N-by-N matrix",
            "negative_reference_definition": (
                "same variant operator on the sampled directed non-edge graph; "
                "A1 represents the selected undirected non-edge pairs bidirectionally"
            ),
            "return_mass_definition": (
                "diag(B^2) from reciprocal sparse edges" if variant.code == "A4"
                else "diag(P^2) from reciprocal sparse edges" if variant.code in ("A5", "A6")
                else "not used"
            ),
        }
    if variant.code == "A1":
        return {
            "execution": "two-layer DGL neighbor sampling with fanout 8",
            "degree_source": "full processed-graph in-degree on sampled one-hop edges",
            "two_hop_definition": "not applicable",
            "negative_reference_definition": (
                "random non-target nodes represented as a bidirected synthetic star "
                "with target degree equal to fanout and source degree one"
            ),
            "return_mass_definition": "not used",
        }
    if variant.code in ("A2", "A4"):
        return {
            "execution": "two-layer DGL neighbor sampling with fanout 8",
            "degree_source": "sampled path counts in the two current DGL blocks",
            "two_hop_definition": "sum over sampled two-hop paths in the current blocks",
            "negative_reference_definition": "random non-target one/two-hop contexts with the same raw sums",
            "return_mass_definition": (
                "count of sampled reciprocal two-hop paths returning to the target"
                if variant.code == "A4" else "not used"
            ),
        }
    return {
        "execution": "two-layer DGL neighbor sampling with fanout 8",
        "degree_source": (
            "sampled block in-degree for row means; full processed-graph in-degree "
            "for the stored return-probability vector"
        ),
        "two_hop_definition": "recursive sampled row means across the two current blocks",
        "negative_reference_definition": "random non-target one/two-hop contexts with the same row-mean arithmetic",
        "return_mass_definition": (
            "full processed-graph diag(P^2) applied to the sampled two-hop response"
            if variant.code in ("A5", "A6") else "not used"
        ),
    }
