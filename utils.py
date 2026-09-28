"""Small graph-tensor utilities shared by loaders and trainers."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def scipy_to_torch_sparse(matrix: sp.spmatrix) -> torch.Tensor:
    matrix = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((matrix.row, matrix.col)).astype(np.int64, copy=False)
    )
    values = torch.from_numpy(matrix.data)
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()


def normalized_laplacian_with_self_loop(
    adjacency: torch.Tensor, eps: float = 1e-12,
) -> torch.Tensor:
    """Return ``I - D^-1/2 (A + I) D^-1/2`` for dense or sparse A."""
    if not adjacency.is_sparse:
        size = adjacency.size(0)
        identity = torch.eye(size, dtype=adjacency.dtype, device=adjacency.device)
        with_self = adjacency + identity
        degree = with_self.sum(dim=1).clamp_min(eps)
        scale = degree.pow(-0.5)
        normalized = scale.unsqueeze(1) * with_self * scale.unsqueeze(0)
        return identity - normalized

    adjacency = adjacency.coalesce()
    size = adjacency.size(0)
    nodes = torch.arange(size, dtype=torch.long, device=adjacency.device)
    diagonal = torch.stack((nodes, nodes), dim=0)
    ones = torch.ones(size, dtype=adjacency.dtype, device=adjacency.device)
    with_self = torch.sparse_coo_tensor(
        torch.cat((adjacency.indices(), diagonal), dim=1),
        torch.cat((adjacency.values(), ones), dim=0),
        adjacency.size(),
        dtype=adjacency.dtype,
        device=adjacency.device,
    ).coalesce()
    indices = with_self.indices()
    degree = torch.sparse.sum(with_self, dim=1).to_dense().clamp_min(eps)
    scale = degree.pow(-0.5)
    normalized_values = with_self.values() * scale[indices[0]] * scale[indices[1]]
    return torch.sparse_coo_tensor(
        torch.cat((diagonal, indices), dim=1),
        torch.cat((ones, -normalized_values), dim=0),
        adjacency.size(),
        dtype=adjacency.dtype,
        device=adjacency.device,
    ).coalesce()


def negative_sampling(adjacency: torch.Tensor) -> torch.Tensor:
    """Sample the historical directed non-edge graph used by alpha training."""
    adjacency = adjacency.coalesce()
    device = adjacency.device
    indices = adjacency.indices()
    num_nodes = adjacency.size(0)
    num_samples = indices.size(1)
    existing = torch.unique(indices[0] * num_nodes + indices[1], sorted=True)
    negative = torch.empty((2, num_samples), dtype=torch.long, device=device)

    filled = 0
    while filled < num_samples:
        remaining = num_samples - filled
        candidate_count = max(remaining * 2, 1024)
        source = torch.randint(0, num_nodes, (candidate_count,), device=device)
        destination = torch.randint(0, num_nodes, (candidate_count,), device=device)
        upper = source < destination
        source, destination = source[upper], destination[upper]
        encoded = source * num_nodes + destination
        positions = torch.searchsorted(existing, encoded)
        valid = positions < existing.numel()
        is_new = torch.ones_like(valid)
        is_new[valid] = existing[positions[valid]] != encoded[valid]
        source, destination = source[is_new], destination[is_new]
        take = min(source.numel(), remaining)
        if take:
            negative[:, filled:filled + take] = torch.stack(
                (source[:take], destination[:take]), dim=0
            )
            filled += take

    values = torch.ones(num_samples, dtype=adjacency.dtype, device=device)
    return torch.sparse_coo_tensor(
        negative, values, adjacency.size(), dtype=adjacency.dtype, device=device
    ).coalesce()


def bidirect_unweighted(adjacency: torch.Tensor) -> torch.Tensor:
    """Represent a sampled undirected edge set in both sparse directions."""
    adjacency = adjacency.coalesce()
    indices = adjacency.indices()
    bidirected = torch.sparse_coo_tensor(
        torch.cat((indices, indices.flip(0)), dim=1),
        torch.ones(
            indices.size(1) * 2,
            dtype=adjacency.dtype,
            device=adjacency.device,
        ),
        adjacency.size(),
        dtype=adjacency.dtype,
        device=adjacency.device,
    ).coalesce()
    return torch.sparse_coo_tensor(
        bidirected.indices(),
        torch.ones_like(bidirected.values()),
        bidirected.size(),
        dtype=bidirected.dtype,
        device=bidirected.device,
    ).coalesce()
