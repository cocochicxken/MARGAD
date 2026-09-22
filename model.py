"""Neural components for MARGAD's three complementary anomaly signals.

The historical code names ``alpha`` and ``gamma`` correspond to TA-DiffRef
and WaveShift, respectively; they are retained for checkpoint compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWaveletAffinity(nn.Module):
    """Implement TA-DiffRef, the target-anonymized multiscale affinity branch.

    With ``P = D^{-1}(A - diag(A))`` and ``L_rw = I - P``, the one- and
    two-hop contexts are respectively ``P H`` and an anonymised ``P^2 H``.
    The latter removes the target's return-path coefficient before being
    re-normalised, so neither context contains a direct copy of ``H_i``.
    Both contexts remain row-stochastic neighbour averages and can therefore
    be compared to ``H_i`` without the scale mismatch of a signed wavelet
    response.
    """

    VALID_MODES = (
        "full",
        "one_hop",
        "anonymous_two_hop",
        "fixed_equal_multiscale",
        "learned_no_coefficient_normalization",
        "learned_no_target_anonymization",
        "learned_no_degree_or_path_volume_normalization",
        "paper_matched_volume",
    )

    def __init__(self, eps=1e-12, mode="full"):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported alpha mode: {mode}")
        self.mode = mode
        # [two-hop, one-hop] logits.  They are equivalent after softmax to
        # [2, 0], [1, 1], and [0, 2] up to per-row constant offsets.
        fixed_coefficients = {
            "one_hop": (0.0, 1.0),
            "anonymous_two_hop": (1.0, 0.0),
            "fixed_equal_multiscale": (0.5, 0.5),
        }
        if mode in fixed_coefficients:
            self.register_parameter("theta_logits", None)
            self.register_buffer(
                "fixed_coefficients",
                torch.tensor([fixed_coefficients[mode]] * 3, dtype=torch.float32),
            )
        else:
            self.theta_logits = nn.Parameter(torch.tensor([
                [1.1, 0.9],
                [0.1, 1.0],
                [1.0, 0.1],
            ], dtype=torch.float32))
            self.register_buffer("fixed_coefficients", None)
        self.eps = eps

    @staticmethod
    def _adj_mm(adj, features):
        if adj.is_sparse:
            return torch.sparse.mm(adj, features)
        return torch.matmul(adj, features)

    @staticmethod
    def _row_sum(adj):
        if adj.is_sparse:
            return torch.sparse.sum(adj, dim=1).to_dense()
        return adj.sum(dim=1)

    @staticmethod
    def _remove_diagonal(adj):
        """Remove explicit self edges without changing the caller's graph."""
        if not adj.is_sparse:
            return adj - torch.diag_embed(torch.diagonal(adj))
        adj = adj.coalesce()
        indices = adj.indices()
        keep = indices[0] != indices[1]
        return torch.sparse_coo_tensor(
            indices[:, keep],
            adj.values()[keep],
            adj.size(),
            dtype=adj.dtype,
            device=adj.device,
        ).coalesce()

    def _random_walk_operator(self, adj, dtype, assume_no_return=False):
        """Return P and diag(P^2), without ever materialising P^2."""
        adj = self._remove_diagonal(adj).to(dtype=dtype)
        degree = self._row_sum(adj)
        if adj.is_sparse:
            adj = adj.coalesce()
            indices = adj.indices()
            values = adj.values() / degree[indices[0]].clamp_min(self.eps)
            walk = torch.sparse_coo_tensor(
                indices, values, adj.size(), dtype=dtype, device=adj.device
            ).coalesce()
        else:
            walk = adj / degree.clamp_min(self.eps).unsqueeze(-1)
        if assume_no_return:
            return walk, torch.zeros(
                walk.size(0), dtype=dtype, device=walk.device
            )
        return walk, self._two_hop_return_probability(walk)

    @staticmethod
    def _two_hop_return_probability(walk):
        """Compute ``diag(P^2)_i = sum_j P_ij P_ji`` in O(|E|) memory.

        Sparse COO tensors are coalesced in lexicographic index order, which
        lets us find each reverse edge by binary search rather than forming a
        potentially dense two-hop matrix.
        """
        if not walk.is_sparse:
            return (walk * walk.transpose(0, 1)).sum(dim=1)

        walk = walk.coalesce()
        indices, values = walk.indices(), walk.values()
        num_nodes = walk.size(0)
        if values.numel() == 0:
            return torch.zeros(num_nodes, dtype=values.dtype, device=values.device)

        reverse = walk.transpose(0, 1).coalesce()
        reverse_indices, reverse_values = reverse.indices(), reverse.values()
        keys = indices[0] * num_nodes + indices[1]
        reverse_keys = reverse_indices[0] * num_nodes + reverse_indices[1]
        positions = torch.searchsorted(reverse_keys, keys)
        valid = positions < reverse_keys.numel()
        reverse_at_edge = torch.zeros_like(values)
        if valid.any():
            valid_positions = positions[valid]
            matched = reverse_keys[valid_positions] == keys[valid]
            if matched.any():
                source_positions = torch.nonzero(valid, as_tuple=False).squeeze(-1)[matched]
                reverse_at_edge[source_positions] = reverse_values[valid_positions[matched]]
        return torch.zeros(num_nodes, dtype=values.dtype, device=values.device).scatter_add(
            0, indices[0], values * reverse_at_edge
        )

    def random_walk_terms(self, adj, dtype, assume_no_return=False):
        """Precompute the graph-only terms reused by full-graph alpha."""
        return self._random_walk_operator(adj, dtype, assume_no_return)

    def raw_path_terms(self, adj, dtype):
        """Raw A/A² propagation and volumes without materialising A².

        This is intentionally used only by A5/A6.  The standard ``full``
        mode remains the target-anonymised random-walk implementation above.
        """
        raw = self._remove_diagonal(adj).to(dtype=dtype)
        one_volume = self._row_sum(raw)
        two_volume = self._adj_mm(raw, one_volume.unsqueeze(-1)).squeeze(-1)
        return raw, one_volume, two_volume

    def precompute_terms(self, adj, dtype, assume_no_return=False):
        """Select the graph-only terms required by the configured mode."""
        if self.mode in (
            "learned_no_degree_or_path_volume_normalization",
            "paper_matched_volume",
        ):
            return self.raw_path_terms(adj, dtype)
        return self.random_walk_terms(adj, dtype, assume_no_return=assume_no_return)

    @property
    def uses_raw_path_volume(self):
        return self.mode in (
            "learned_no_degree_or_path_volume_normalization",
            "paper_matched_volume",
        )

    @property
    def requires_two_hop(self):
        return self.mode != "one_hop"

    @property
    def uses_target_anonymization(self):
        return self.mode in (
            "full",
            "anonymous_two_hop",
            "fixed_equal_multiscale",
            "learned_no_coefficient_normalization",
        )

    def filter_coefficients(self, dtype=None):
        """Return the three final [two-hop, one-hop] alpha coefficients."""
        if self.fixed_coefficients is not None:
            coefficients = self.fixed_coefficients
            return coefficients if dtype is None else coefficients.to(dtype=dtype)
        coefficients = F.softmax(self.theta_logits, dim=1)
        offsets = torch.zeros_like(coefficients)
        offsets[1, 1] = 1.0
        offsets[2, 0] = 1.0
        coefficients = coefficients + offsets
        return coefficients if dtype is None else coefficients.to(dtype=dtype)

    def mixing_weights(self, dtype=None):
        """Normalise each positive one/two-hop mixture to a convex average."""
        coefficients = self.filter_coefficients(dtype=dtype)
        if self.mode == "learned_no_coefficient_normalization":
            return coefficients
        return coefficients / coefficients.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def diagnostic_coefficients(self, dtype=None):
        """Expose both learned/fixed coefficients and actual mixing weights."""
        return {
            "mode": self.mode,
            "coefficient_normalized": self.mode != "learned_no_coefficient_normalization",
            "filter_coefficients": self.filter_coefficients(dtype=dtype),
            "mixing_weights": self.mixing_weights(dtype=dtype),
        }

    def _anonymous_two_hop(self, two_hop, one_hop, features, return_probability):
        remaining_mass = 1.0 - return_probability
        anonymous = (
            two_hop - return_probability.unsqueeze(-1) * features
        ) / remaining_mass.clamp_min(self.eps).unsqueeze(-1)
        return torch.where(
            (remaining_mass > self.eps).unsqueeze(-1), anonymous, one_hop
        )

    def combine_sampled_terms(
        self,
        features,
        one_hop,
        two_hop=None,
        return_probability=None,
        one_volume=None,
        two_volume=None,
        return_filters=False,
    ):
        """Combine precomputed full-graph or sampled alpha messages.

        ``one_volume``/``two_volume`` are only meaningful for raw A/A² modes.
        In T-Social they are fanout-sampled path counts, not exact full-graph
        volumes; callers preserve that distinction in their diagnostics.
        """
        if self.mode == "one_hop":
            two_hop = one_hop if two_hop is None else two_hop
        elif two_hop is None:
            raise ValueError(f"Alpha mode {self.mode} requires a two-hop message.")

        if self.uses_target_anonymization:
            if return_probability is None:
                raise ValueError("Target-anonymised alpha requires return probabilities.")
            two_hop = self._anonymous_two_hop(
                two_hop, one_hop, features, return_probability
            )

        coefficients = self.mixing_weights(dtype=features.dtype)
        h_near_per_filter = []
        for two_hop_coeff, one_hop_coeff in coefficients:
            if self.mode == "paper_matched_volume":
                if one_volume is None or two_volume is None:
                    raise ValueError("paper_matched_volume requires A/A² volumes.")
                numerator = two_hop_coeff * two_hop + one_hop_coeff * one_hop
                denominator = (
                    two_hop_coeff * two_volume + one_hop_coeff * one_volume
                ).clamp_min(self.eps).unsqueeze(-1)
                h_near_per_filter.append(numerator / denominator)
            else:
                h_near_per_filter.append(
                    two_hop_coeff * two_hop + one_hop_coeff * one_hop
                )

        h_near_per_filter = torch.stack(h_near_per_filter, dim=0)
        h_near = h_near_per_filter.mean(dim=0)
        if return_filters:
            return h_near, h_near_per_filter
        return h_near

    def forward(
        self, features, adj, walk_terms=None, return_filters=False,
    ):
        if self.uses_raw_path_volume:
            if walk_terms is None:
                raw, one_volume, two_volume = self.raw_path_terms(adj, features.dtype)
            else:
                raw, one_volume, two_volume = walk_terms
            one_hop = self._adj_mm(raw, features)
            two_hop = self._adj_mm(raw, one_hop)
            return self.combine_sampled_terms(
                features,
                one_hop,
                two_hop,
                one_volume=one_volume,
                two_volume=two_volume,
                return_filters=return_filters,
            )

        if walk_terms is None:
            walk, return_probability = self.random_walk_terms(adj, features.dtype)
        else:
            walk, return_probability = walk_terms
        one_hop = self._adj_mm(walk, features)
        two_hop = self._adj_mm(walk, one_hop) if self.requires_two_hop else None
        return self.combine_sampled_terms(
            features,
            one_hop,
            two_hop,
            return_probability=return_probability,
            return_filters=return_filters,
        )


class MultiFilterGammaWavelet(nn.Module):
    """Implement WaveShift with shared and channel-specific spectral responses."""

    VALID_MODES = (
        "full",
        "fixed_shared_wavelet",
        "learned_global_shared_only",
        "learned_channel_specific_only",
        "dual_independent",
        "uncentered_cross_response",
    )

    def __init__(self, hidden_size, init_std=0.02, mode="full"):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported gamma mode: {mode}")
        self.mode = mode
        # Coefficient order: [L^2 H, L H, H].
        templates = torch.tensor([
            [0.25, -1.0, 1.0],
            [-1.0, 2.0, 0.0],
            [0.25, 0.0, 0.0],
        ], dtype=torch.float32)
        self.register_buffer('beta_templates', templates)
        self.hidden_size = hidden_size

        if mode == "fixed_shared_wavelet":
            self.register_parameter("thick_coefficients", None)
            self.register_parameter("thin_coefficients", None)
            return

        if mode in ("full", "learned_global_shared_only", "dual_independent", "uncentered_cross_response"):
            self.thick_coefficients = nn.Parameter(torch.zeros_like(templates))
        else:
            self.register_parameter("thick_coefficients", None)

        if mode in ("full", "learned_channel_specific_only", "dual_independent", "uncentered_cross_response"):
            # Thin retained the Beta-wavelet basis at this stage.  Channel
            # groups receive different initial spectral offsets to avoid
            # channel sharing.  The full-mode branch is deliberately the
            # historical initialization sequence.
            channel_template_ids = torch.arange(hidden_size) % 3
            group_templates = templates.unsqueeze(1).expand(-1, hidden_size, -1)
            channel_templates = templates[channel_template_ids].unsqueeze(0)
            thin_init = (
                0.2 * (channel_templates - group_templates)
                + init_std * torch.randn(3, hidden_size, 3)
            )
            self.thin_coefficients = nn.Parameter(thin_init)
        else:
            self.register_parameter("thin_coefficients", None)

    @staticmethod
    def _matmul(laplacian, features):
        if laplacian.is_sparse:
            return torch.sparse.mm(laplacian, features)
        return torch.matmul(laplacian, features)

    def effective_coefficients(self, squash_with_tanh=True):
        thick = (
            self.beta_templates + self.thick_coefficients
            if self.thick_coefficients is not None else None
        )
        thin = (
            self.beta_templates.unsqueeze(1) + self.thin_coefficients
            if self.thin_coefficients is not None else None
        )
        if self.mode == "fixed_shared_wavelet":
            thick = self.beta_templates
        if squash_with_tanh:
            thick = torch.tanh(thick) if thick is not None else None
            thin = torch.tanh(thin) if thin is not None else None
        return thick, thin

    def response_curves(self, lambdas, squash_with_tanh=True):
        """Evaluate g(lambda)=a2 lambda²+a1 lambda+a0 for diagnostics."""
        lambdas = torch.as_tensor(
            lambdas, dtype=self.beta_templates.dtype, device=self.beta_templates.device
        )
        thick, thin = self.effective_coefficients(squash_with_tanh)

        def response(coefficients):
            if coefficients is None:
                return None
            return (
                coefficients[..., 0].unsqueeze(-1) * lambdas.square()
                + coefficients[..., 1].unsqueeze(-1) * lambdas
                + coefficients[..., 2].unsqueeze(-1)
            )

        return response(thick), response(thin)

    def diagnostic_coefficients(self, squash_with_tanh=True):
        thick, thin = self.effective_coefficients(squash_with_tanh)
        return {
            "mode": self.mode,
            "shared_coefficients": thick,
            "channel_specific_coefficients": thin,
        }

    def forward(self, h, laplacian, squash_coefficients=True):
        lh = self._matmul(laplacian, h)
        l2h = self._matmul(laplacian, lh)
        thick_outputs, thin_outputs = [], []
        thick_coefficients, thin_coefficients = self.effective_coefficients(
            squash_with_tanh=squash_coefficients
        )
        for filter_id in range(3):
            if thick_coefficients is not None:
                thick_coeff = thick_coefficients[filter_id]
                thick_outputs.append(
                    thick_coeff[0] * l2h + thick_coeff[1] * lh + thick_coeff[2] * h
                )

            if thin_coefficients is not None:
                thin_coeff = thin_coefficients[filter_id]
                thin_outputs.append(
                    l2h * thin_coeff[:, 0]
                    + lh * thin_coeff[:, 1]
                    + h * thin_coeff[:, 2]
                )
        thick = torch.stack(thick_outputs, dim=0) if thick_outputs else None
        thin = torch.stack(thin_outputs, dim=0) if thin_outputs else None
        return thick, thin


class GAD(nn.Module):
    """Combine the shared encoder with TA-DiffRef and WaveShift modules."""

    def __init__(
        self,
        feat_size,
        hidden_size,
        dropout,
        alpha_mode="full",
        gamma_mode="full",
    ):
        super().__init__()
        self.lin = nn.Linear(feat_size, hidden_size)
        self.alpha_wavelet = AdaptiveWaveletAffinity(mode=alpha_mode)
        self.gamma_wavelet = MultiFilterGammaWavelet(hidden_size, mode=gamma_mode)

    def encode(self, x, apply_tanh=True):
        """Encode features, optionally retaining the raw linear gamma signal."""
        x_lin = self.lin(x)
        return torch.tanh(x_lin) if apply_tanh else x_lin

    def forward(self, x, laplacian=None):
        """Encode features and optionally return both WaveShift responses."""

        x_lin = self.encode(x)
        if laplacian is None:
            return x_lin
        thick_bands, thin_bands = self.gamma_wavelet(x_lin, laplacian)
        return x_lin, thick_bands, thin_bands

    def local_affinity(
        self, h, adj, normalize=True, walk_terms=None, return_filter_scores=False,
    ):
        """Compute TA-DiffRef affinity scores for the supplied representations."""

        if normalize:
            h = F.normalize(h, p=2, dim=1)
        if not return_filter_scores:
            h_near = self.alpha_wavelet(h, adj, walk_terms=walk_terms)
            return (h * h_near).sum(dim=1)

        h_near, per_filter_near = self.alpha_wavelet(
            h, adj, walk_terms=walk_terms, return_filters=True
        )
        total_score = (h * h_near).sum(dim=1)
        filter_scores = (h.unsqueeze(0) * per_filter_near).sum(dim=-1).transpose(0, 1)
        return total_score, filter_scores
