"""Load and preprocess the seven datasets supported by the MARGAD code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch

from dataset_config import DatasetSpec
from utils import scipy_to_torch_sparse


@dataclass(frozen=True)
class FullGraphData:
    """In-memory tensors and arrays used by the full-graph trainer."""

    adjacency: torch.Tensor
    features: np.ndarray
    labels: np.ndarray
    evaluation_index: np.ndarray | None = None


@dataclass(frozen=True)
class LargeGraphData:
    """DGL graph and node tensors used by the sampled T-Social trainer."""

    graph: object
    features: torch.Tensor
    labels: torch.Tensor
    evaluation_index: torch.Tensor | None = None


def _prepare_scipy_adjacency(adjacency) -> sp.csr_matrix:
    """Apply MARGAD preprocessing: max-symmetrize and remove explicit loops."""
    adjacency = sp.csr_matrix(adjacency, dtype=np.float32)
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    diagonal = adjacency.diagonal()
    if np.any(diagonal):
        adjacency = adjacency - sp.diags(diagonal, format="csr")
    adjacency.eliminate_zeros()
    return adjacency


def _dense_float_features(features) -> np.ndarray:
    if sp.issparse(features):
        features = features.toarray()
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got {features.shape}.")
    return features


def _finalize_full_graph(
    spec: DatasetSpec,
    adjacency,
    features,
    labels,
    evaluation_index=None,
) -> FullGraphData:
    adjacency = _prepare_scipy_adjacency(adjacency)
    features = _dense_float_features(features)
    labels = np.asarray(labels).reshape(-1)
    if features.shape[0] != adjacency.shape[0] or labels.size != adjacency.shape[0]:
        raise ValueError(
            "Adjacency, features, and labels must contain the same node count: "
            f"{adjacency.shape[0]}, {features.shape[0]}, {labels.size}."
        )
    if spec.standardize_features:
        # torch.std(..., dim=0) historically used the sample standard deviation.
        deviation = features.std(axis=0, ddof=1)
        features = (features - features.mean(axis=0)) / (deviation + 1e-30)
        features = features.astype(np.float32, copy=False)
    if evaluation_index is not None:
        evaluation_index = np.asarray(evaluation_index, dtype=np.int64).reshape(-1)
        if not evaluation_index.size:
            raise ValueError(f"{spec.cli_name} has no labeled evaluation nodes.")
    return FullGraphData(
        scipy_to_torch_sparse(adjacency),
        features,
        labels,
        evaluation_index,
    )


def _mat_field(payload: dict, names: tuple[str, ...], description: str):
    for name in names:
        if name in payload:
            return payload[name]
    lower_names = {key.lower(): key for key in payload if not key.startswith("__")}
    for name in names:
        actual = lower_names.get(name.lower())
        if actual is not None:
            return payload[actual]
    available = ", ".join(sorted(key for key in payload if not key.startswith("__")))
    raise KeyError(f"Missing {description}; tried {names}. Available fields: {available}")


def _load_mat(spec: DatasetSpec, data_dir: Path) -> FullGraphData:
    path = data_dir / f"{spec.cli_name}.mat"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    payload = sio.loadmat(path)
    adjacency = _mat_field(payload, ("Network", "A", "homo", "adj"), "adjacency")
    features = _mat_field(
        payload, ("Attributes", "X", "features", "feature", "feat"), "features"
    )
    labels = _mat_field(payload, ("Label", "gnd", "label", "labels", "y"), "labels")
    return _finalize_full_graph(spec, adjacency, features, labels)


def _load_elliptic(spec: DatasetSpec, data_dir: Path) -> FullGraphData:
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError("Elliptic loading requires pandas.") from error

    root = data_dir / "elliptic" / "elliptic_bitcoin_dataset"
    feature_path = root / "elliptic_txs_features.csv"
    class_path = root / "elliptic_txs_classes.csv"
    edge_path = root / "elliptic_txs_edgelist.csv"
    for path in (feature_path, class_path, edge_path):
        if not path.is_file():
            raise FileNotFoundError(f"Elliptic file not found: {path}")

    feature_frame = pd.read_csv(feature_path, header=None)
    transaction_ids = feature_frame.iloc[:, 0].to_numpy(dtype=np.int64, copy=False)
    features = feature_frame.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
    classes = pd.read_csv(class_path, dtype={"txId": np.int64, "class": str})
    class_by_id = classes.set_index("txId")["class"].reindex(transaction_ids)
    known = class_by_id.isin(("1", "2")).to_numpy()
    labels = (class_by_id == "1").to_numpy(dtype=np.int64)
    evaluation_index = np.flatnonzero(known)

    edges = pd.read_csv(edge_path, dtype={"txId1": np.int64, "txId2": np.int64})
    node_index = pd.Series(np.arange(transaction_ids.size, dtype=np.int64), index=transaction_ids)
    source = edges["txId1"].map(node_index)
    destination = edges["txId2"].map(node_index)
    valid = source.notna() & destination.notna()
    source = source.loc[valid].to_numpy(dtype=np.int64, copy=False)
    destination = destination.loc[valid].to_numpy(dtype=np.int64, copy=False)
    adjacency = sp.coo_matrix(
        (np.ones(source.size, dtype=np.float32), (source, destination)),
        shape=(transaction_ids.size, transaction_ids.size),
    )
    return _finalize_full_graph(
        spec, adjacency, features, labels, evaluation_index
    )


def _dgl_field(graph, names: tuple[str, ...], description: str):
    for name in names:
        if name in graph.ndata:
            return graph.ndata[name]
    raise KeyError(
        f"DGL graph is missing {description}; tried {names}, "
        f"available={sorted(graph.ndata.keys())}."
    )


def _raw_dgl_path(data_dir: Path, dataset_key: str) -> Path:
    path = data_dir / dataset_key / dataset_key
    if path.is_file():
        return path
    fallback = data_dir / dataset_key
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"DGL graph not found: {path}")


def _process_dgl_graph(graph):
    import dgl

    features = _dgl_field(graph, ("feature", "features", "feat", "x"), "features").float()
    labels = _dgl_field(graph, ("label", "labels", "y"), "labels")
    if labels.dim() > 1:
        labels = labels.argmax(dim=-1)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph.ndata["feature"] = features.float().cpu().contiguous()
    graph.ndata["label"] = labels.long().reshape(-1).cpu()
    return graph


def _load_tfinance(spec: DatasetSpec, data_dir: Path) -> FullGraphData:
    try:
        import dgl
    except ImportError as error:
        raise ImportError("T-Finance loading requires DGL.") from error
    graphs, _ = dgl.load_graphs(str(_raw_dgl_path(data_dir, spec.key)))
    if not graphs:
        raise ValueError("T-Finance file did not contain a graph.")
    graph = _process_dgl_graph(graphs[0])
    features = graph.ndata["feature"].numpy()
    labels = graph.ndata["label"].numpy()
    source, destination = graph.edges(order="eid")
    adjacency = sp.coo_matrix(
        (np.ones(source.numel(), dtype=np.float32), (source.numpy(), destination.numpy())),
        shape=(graph.num_nodes(), graph.num_nodes()),
    )
    return _finalize_full_graph(spec, adjacency, features, labels)


def _load_tsocial(spec: DatasetSpec, data_dir: Path) -> LargeGraphData:
    try:
        import dgl
    except ImportError as error:
        raise ImportError("T-Social loading requires DGL.") from error

    raw_path = _raw_dgl_path(data_dir, spec.key)
    cache_path = raw_path.parent / "tsocial__processed_v1.bin"
    cache_fresh = cache_path.is_file() and cache_path.stat().st_mtime >= raw_path.stat().st_mtime
    graph = None
    if cache_fresh:
        graphs, _ = dgl.load_graphs(str(cache_path))
        if graphs:
            graph = graphs[0]
            print(f"Using cached processed T-Social graph: {cache_path.name}")
    if graph is None:
        graphs, _ = dgl.load_graphs(str(raw_path))
        if not graphs:
            raise ValueError("T-Social file did not contain a graph.")
        graph = _process_dgl_graph(graphs[0])
        dgl.save_graphs(str(cache_path), [graph])
        print(f"Built processed T-Social graph: {cache_path.name}")

    # Old compatible caches may not use the canonical aliases yet.
    features = _dgl_field(graph, ("feature", "features", "feat", "x"), "features").float().cpu().contiguous()
    labels = _dgl_field(graph, ("label", "labels", "y"), "labels")
    if labels.dim() > 1:
        labels = labels.argmax(dim=-1)
    graph.ndata["feature"] = features
    graph.ndata["label"] = labels.long().reshape(-1).cpu()
    return LargeGraphData(graph, features, graph.ndata["label"])


def load_full_graph(spec: DatasetSpec, data_dir: str | Path) -> FullGraphData:
    """Load a dataset routed to the full-graph training implementation."""

    root = Path(data_dir).resolve()
    if spec.loader == "mat":
        return _load_mat(spec, root)
    if spec.loader == "elliptic_csv":
        return _load_elliptic(spec, root)
    if spec.loader == "dgl_full":
        return _load_tfinance(spec, root)
    raise ValueError(f"{spec.cli_name} does not use the full-graph loader.")


def load_large_graph(spec: DatasetSpec, data_dir: str | Path) -> LargeGraphData:
    """Load T-Social for the neighbor-sampled training implementation."""

    if spec.loader != "dgl_large":
        raise ValueError(f"{spec.cli_name} does not use the large-graph loader.")
    return _load_tsocial(spec, Path(data_dir).resolve())
