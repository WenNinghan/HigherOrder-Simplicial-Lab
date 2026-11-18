"""Pure PyTorch Higher-Order Simplicial Neural Networks & Baselines.

Implements:
- ``HodgeSimplicialNN``: Decoupled lower (L1_down), upper (L1_up), and harmonic/self
  simplicial convolutional network with exact orientation equivariance ``f(-x) = -f(x)``.
- ``NodeGCNEdgePredictor``: Node-level L0 GCN with antisymmetric potential difference
  readout ``h_j - h_i`` (provably blind to divergence-free curl & harmonic flows).
- ``EdgeLineGraphGCN``: Undirected line-graph / conflated isotropic L1 baseline.
- ``EdgeMLP``: Topology-agnostic edge-wise multilayer perceptron.
- Benchmark datasets & trainers for (1) Missing Edge Flux Imputation & Denoising and
  (2) Topological Homotopy Trajectory Classification around cavities.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .complexes import SimplicialComplex, _canonical_edge
from .hodge import (
    graph_laplacian_0,
    harmonic_basis,
    hodge_laplacian_1,
    synthesize_hodge_flow,
)


def _spectral_normalize(mat: np.ndarray) -> np.ndarray:
    """Normalize a symmetric positive semi-definite matrix by its largest eigenvalue in float64."""
    if mat.size == 0:
        return mat.astype(np.float64)
    evals = np.linalg.eigvalsh(mat)
    lmax = float(np.max(np.abs(evals))) if len(evals) > 0 else 1.0
    scale = max(lmax, 1.0)
    return (mat / scale).astype(np.float64)


def _build_boundary_cycle_cochains(
    sc: SimplicialComplex,
    edge_orientations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Construct closed 1-cycle indicator cochains along inner hole boundaries (or winding 1-forms).

    Transforms covariantly as ``C' = D @ C`` under arbitrary edge orientation flips
    ``D = diag(+-1)``, making ``C.T @ h`` strictly gauge-invariant across all backbones.
    """
    n1 = sc.num_edges
    loops = sc.metadata.get("hole_boundary_loops", [])
    if n1 == 0:
        return np.zeros((0, 0), dtype=np.float64)

    if loops:
        c_mat = np.zeros((n1, len(loops)), dtype=np.float64)
        for m, loop in enumerate(loops):
            cyc = list(loop) + [loop[0]]
            for u, v in zip(cyc[:-1], cyc[1:]):
                ek = _canonical_edge(u, v)
                eidx = sc.edge_to_idx[ek]
                c_mat[eidx, m] += 1.0 if int(u) < int(v) else -1.0
            norm_m = np.linalg.norm(c_mat[:, m])
            if norm_m > 1e-12:
                c_mat[:, m] /= norm_m
    elif sc.coords is not None:
        from .hodge import edge_winding_operator

        c_mat = edge_winding_operator(sc)
    else:
        return np.zeros((n1, 0), dtype=np.float64)

    if edge_orientations is not None:
        signs = np.asarray(edge_orientations, dtype=np.float64).reshape(-1)
        c_mat = signs[:, None] * c_mat
    return c_mat


def build_torch_operators(
    sc: SimplicialComplex,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    edge_orientations: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """Precompute normalized boundary, Laplacian, line-graph, harmonic, and cycle tensors.

    Supports arbitrary per-edge orientation gauge ``edge_orientations`` in ``{-1, +1}^{N1}``
    (``B1' = B1 @ D``, ``B2' = D @ B2``) while preserving full float64 precision before
    casting to ``dtype``.
    """
    if device is None:
        device = torch.device("cpu")

    b1_raw = sc.B1.copy()
    b2_raw = sc.B2.copy()
    signs: Optional[np.ndarray] = None
    if edge_orientations is not None:
        signs = np.asarray(edge_orientations, dtype=np.float64).reshape(-1)
        if signs.shape[0] != sc.num_edges or not np.all(np.isin(np.abs(signs), [1.0])):
            raise ValueError("edge_orientations must have shape (N1,) with entries in {-1, +1}.")
        b1_raw = b1_raw * signs[None, :]
        if sc.num_triangles > 0:
            b2_raw = signs[:, None] * b2_raw

    l0_np = b1_raw @ b1_raw.T
    l1_down_np = b1_raw.T @ b1_raw
    l1_up_np = (b2_raw @ b2_raw.T) if sc.num_triangles > 0 else np.zeros_like(l1_down_np)
    l1_np = l1_down_np + l1_up_np

    l0_norm = _spectral_normalize(l0_np)
    l1_down_norm = _spectral_normalize(l1_down_np)
    l1_up_norm = _spectral_normalize(l1_up_np)
    l1_iso_norm = _spectral_normalize(l1_np)

    # Unsigned line-graph normalized adjacency (orientation-blind edge connectivity)
    lg_adj = np.abs(l1_down_np - np.diag(np.diag(l1_down_np)))
    deg_lg = np.sum(lg_adj, axis=1)
    inv_sqrt = np.where(deg_lg > 0, 1.0 / np.sqrt(np.maximum(deg_lg, 1e-8)), 0.0)
    lg_norm = inv_sqrt[:, None] * lg_adj * inv_sqrt[None, :]

    u_harm_np = harmonic_basis(sc, edge_orientations=signs)
    c_bnd_np = _build_boundary_cycle_cochains(sc, edge_orientations=signs)

    return {
        "B1": torch.from_numpy(b1_raw).to(device=device, dtype=dtype),
        "B2": torch.from_numpy(b2_raw).to(device=device, dtype=dtype),
        "L0_norm": torch.from_numpy(l0_norm).to(device=device, dtype=dtype),
        "L1_down_norm": torch.from_numpy(l1_down_norm).to(device=device, dtype=dtype),
        "L1_up_norm": torch.from_numpy(l1_up_norm).to(device=device, dtype=dtype),
        "L1_iso_norm": torch.from_numpy(l1_iso_norm).to(device=device, dtype=dtype),
        "LG_norm": torch.from_numpy(lg_norm).to(device=device, dtype=dtype),
        "U_harm": torch.from_numpy(u_harm_np).to(device=device, dtype=dtype),
        "C_bnd": torch.from_numpy(c_bnd_np).to(device=device, dtype=dtype),
    }


def _format_edge_input(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    """Standardize edge input tensor shapes to ``(batch, N1, F_in)``."""
    if x.ndim == 1:
        x_3d = x.unsqueeze(0).unsqueeze(-1)
        orig_mode = "1d"
    elif x.ndim == 2:
        # Interpret (N1, F_in) as single complex with F_in channels
        x_3d = x.unsqueeze(0)
        orig_mode = "2d"
    elif x.ndim == 3:
        x_3d = x
        orig_mode = "3d"
    else:
        raise ValueError(f"Expected 1D, 2D, or 3D edge input tensor, got shape {tuple(x.shape)}.")

    m_3d: Optional[torch.Tensor] = None
    if mask is not None:
        if mask.ndim == 1:
            m_3d = mask.unsqueeze(0).unsqueeze(-1)
        elif mask.ndim == 2:
            if mask.shape == x_3d.shape[:2]:
                m_3d = mask.unsqueeze(-1)
            else:
                m_3d = mask.unsqueeze(0)
        elif mask.ndim == 3:
            m_3d = mask
        m_3d = m_3d.to(dtype=x_3d.dtype, device=x_3d.device)

    return x_3d, m_3d, orig_mode


def _restore_edge_output(y_3d: torch.Tensor, orig_mode: str) -> torch.Tensor:
    if orig_mode == "1d":
        return y_3d.squeeze(0).squeeze(-1)
    if orig_mode == "2d":
        return y_3d.squeeze(0)
    return y_3d


def _op_mul(op: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Fast single-GEMM operator application: (M, N) x (B, N, C) -> (B, M, C)."""
    bsz, n_in, ch = h.shape
    flat = h.permute(1, 0, 2).reshape(n_in, bsz * ch)
    out = torch.mm(op, flat)
    return out.reshape(op.shape[0], bsz, ch).permute(1, 0, 2).contiguous()


class HodgeSimplicialConvLayer(nn.Module):
    """Decoupled multi-hop Hodge 1-cochain convolutional layer.

    Computes:
    ``Z = H W_0 + sum_{k=1}^K L1_down^k H W_{down,k} + sum_{k=1}^K L1_up^k H W_{up,k}``
    with optional mask-conditioned odd gating (preserving exact ``f(-x) = -f(x)``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_order: int = 2,
        equivariant: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_order = filter_order
        self.equivariant = equivariant
        use_bias = not equivariant

        self.lin_self = nn.Linear(in_channels, out_channels, bias=use_bias)
        self.lin_mask_obs = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_down = nn.ModuleList(
            [nn.Linear(in_channels, out_channels, bias=use_bias) for _ in range(filter_order)]
        )
        self.lin_up = nn.ModuleList(
            [nn.Linear(in_channels, out_channels, bias=use_bias) for _ in range(filter_order)]
        )

    def forward(
        self,
        h: torch.Tensor,
        l1_down: torch.Tensor,
        l1_up: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.lin_self(h)
        if mask is not None:
            out = out + self.lin_mask_obs(mask * h)

        h_down = h
        h_up = h
        for k in range(self.filter_order):
            h_down = _op_mul(l1_down, h_down)
            h_up = _op_mul(l1_up, h_up)
            out = out + self.lin_down[k](h_down) + self.lin_up[k](h_up)
        return out


class HodgeSimplicialNN(nn.Module):
    """Orientation-Equivariant Simplicial Message Passing / Hodge Convolutional Network.

    Decouples lower boundary coboundary diffusion (``L1_down = B1.T @ B1``),
    upper triangle curl diffusion (``L1_up = B2 @ B2.T``), and harmonic/self
    channels. When ``equivariant=True`` (default), biases are disabled and odd
    ``tanh`` nonlinearities guarantee exact local edge-orientation gauge equivariance
    ``F(D x, M; B1 D, D B2) = D F(x, M; B1, B2)`` for all ``D = diag(+-1)``.
    """

    def __init__(
        self,
        sc: SimplicialComplex,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 1,
        num_layers: int = 4,
        filter_order: int = 2,
        equivariant: bool = True,
        edge_orientations: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        ops_f64 = build_torch_operators(
            sc, dtype=torch.float64, edge_orientations=edge_orientations
        )
        self._ops_f64 = {
            k: ops_f64[k] for k in ("L1_down_norm", "L1_up_norm", "U_harm")
        }
        self.register_buffer("L1_down_norm", ops_f64["L1_down_norm"].to(dtype=dtype))
        self.register_buffer("L1_up_norm", ops_f64["L1_up_norm"].to(dtype=dtype))
        self.register_buffer("U_harm", ops_f64["U_harm"].to(dtype=dtype))

        self.equivariant = equivariant
        use_bias = not equivariant
        self.act = torch.tanh if equivariant else nn.GELU()

        self.in_proj = nn.Linear(in_channels, hidden_channels, bias=use_bias)
        self.layers = nn.ModuleList(
            [
                HodgeSimplicialConvLayer(
                    hidden_channels,
                    hidden_channels,
                    filter_order=filter_order,
                    equivariant=equivariant,
                )
                for _ in range(num_layers)
            ]
        )
        self.harm_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)
        self.out_proj = nn.Linear(hidden_channels * 2, out_channels, bias=use_bias)
        if dtype != torch.float32:
            self.to(dtype=dtype)

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "_ops_f64") and self.L1_down_norm.dtype == torch.float64:
            dev = self.L1_down_norm.device
            self.L1_down_norm.copy_(self._ops_f64["L1_down_norm"].to(device=dev))
            self.L1_up_norm.copy_(self._ops_f64["L1_up_norm"].to(device=dev))
            self.U_harm.copy_(self._ops_f64["U_harm"].to(device=dev))
        return self

    def encode_edges(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.act(self.in_proj(x))
        h0 = h
        for layer in self.layers:
            h_next = self.act(layer(h, self.L1_down_norm, self.L1_up_norm, mask=mask))
            h = h + h_next
            if mask is not None:
                # Re-anchor observed edges in latent space while remaining strictly odd in x
                h = h + 0.25 * (mask * h0)

        # Global harmonic subspace projection (U_harm @ U_harm.T) @ h @ W_harm.T
        if self.U_harm.shape[1] > 0:
            harm_coeff = _op_mul(self.U_harm.transpose(0, 1).contiguous(), h)
            h_harm = _op_mul(self.U_harm, self.harm_proj(harm_coeff))
        else:
            h_harm = torch.zeros_like(h)

        return torch.cat([h, h_harm], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_3d, m_3d, orig_mode = _format_edge_input(x, mask)
        feat = self.encode_edges(x_3d, mask=m_3d)
        out = self.out_proj(feat)
        if m_3d is not None and out.shape[-1] == x_3d.shape[-1]:
            # Residual skip on observed edges (strictly odd in x_3d)
            out = m_3d * x_3d + (1.0 - m_3d) * out + 0.15 * (m_3d * (out - x_3d))
        return _restore_edge_output(out, orig_mode)

    def with_edge_orientations(
        self,
        sc: SimplicialComplex,
        edge_orientations: np.ndarray,
    ) -> "HodgeSimplicialNN":
        """Clone learnable parameters onto ``sc`` with gauge-flipped edge orientations ``D = diag(+-1)``."""
        clone = HodgeSimplicialNN(
            sc=sc,
            in_channels=self.in_proj.in_features,
            hidden_channels=self.in_proj.out_features,
            out_channels=self.out_proj.out_features,
            num_layers=len(self.layers),
            filter_order=self.layers[0].filter_order if len(self.layers) > 0 else 2,
            equivariant=self.equivariant,
            edge_orientations=edge_orientations,
            dtype=self.in_proj.weight.dtype,
        ).to(device=self.in_proj.weight.device)
        src_params = dict(self.named_parameters())
        with torch.no_grad():
            for name, p_dst in clone.named_parameters():
                p_dst.copy_(src_params[name])
        if not self.training:
            clone.eval()
        return clone


class NodeGCNEdgePredictor(nn.Module):
    """Node-level L0 GCN followed by antisymmetric edge difference readout ``h_j - h_i``.

    Maps edge observations to vertices via divergence ``B1 @ x``, performs message
    passing with normalized ``L0``, and reconstructs edge flows via ``B1.T @ phi_hat``.
    By ``im(B1.T) perp ker(B1)``, predictions have identically zero curl and harmonic
    components, making this architecture provably blind to solenoidal flows.
    """

    def __init__(
        self,
        sc: SimplicialComplex,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 1,
        num_layers: int = 4,
        equivariant: bool = True,
        edge_orientations: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        ops_f64 = build_torch_operators(
            sc, dtype=torch.float64, edge_orientations=edge_orientations
        )
        self._ops_f64 = {k: ops_f64[k] for k in ("B1", "L0_norm")}
        self.register_buffer("B1", ops_f64["B1"].to(dtype=dtype))
        self.register_buffer("L0_norm", ops_f64["L0_norm"].to(dtype=dtype))

        use_bias = not equivariant
        self.act = torch.tanh if equivariant else nn.GELU()

        self.in_proj = nn.Linear(in_channels, hidden_channels, bias=use_bias)
        self.self_lins = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels, bias=use_bias) for _ in range(num_layers)]
        )
        self.neigh_lins = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels, bias=use_bias) for _ in range(num_layers)]
        )
        self.node_out = nn.Linear(hidden_channels, out_channels, bias=use_bias)
        if dtype != torch.float32:
            self.to(dtype=dtype)

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "_ops_f64") and self.B1.dtype == torch.float64:
            dev = self.B1.device
            self.B1.copy_(self._ops_f64["B1"].to(device=dev))
            self.L0_norm.copy_(self._ops_f64["L0_norm"].to(device=dev))
        return self

    def encode_edges(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Aggregate edge 1-cochain to 0-cochain divergence on vertices: (batch, N0, F_in)
        x_node = _op_mul(self.B1, x)
        h = self.act(self.in_proj(x_node))
        for lin_s, lin_n in zip(self.self_lins, self.neigh_lins):
            lh = _op_mul(self.L0_norm, h)
            h = h + self.act(lin_s(h) + lin_n(lh))
        # Antisymmetric edge difference h_j - h_i = B1.T @ h
        return _op_mul(self.B1.transpose(0, 1).contiguous(), h)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_3d, _, orig_mode = _format_edge_input(x, mask)
        edge_diff = self.encode_edges(x_3d, mask=None)
        # Linear projection preserves exact membership in im(B1.T)
        out = self.node_out(edge_diff)
        return _restore_edge_output(out, orig_mode)

    def with_edge_orientations(
        self,
        sc: SimplicialComplex,
        edge_orientations: np.ndarray,
    ) -> "NodeGCNEdgePredictor":
        """Clone learnable parameters onto ``sc`` with gauge-flipped edge orientations ``D = diag(+-1)``."""
        clone = NodeGCNEdgePredictor(
            sc=sc,
            in_channels=self.in_proj.in_features,
            hidden_channels=self.in_proj.out_features,
            out_channels=self.node_out.out_features,
            num_layers=len(self.self_lins),
            equivariant=self.in_proj.bias is None,
            edge_orientations=edge_orientations,
            dtype=self.in_proj.weight.dtype,
        ).to(device=self.in_proj.weight.device)
        src_params = dict(self.named_parameters())
        with torch.no_grad():
            for name, p_dst in clone.named_parameters():
                p_dst.copy_(src_params[name])
        if not self.training:
            clone.eval()
        return clone


class EdgeLineGraphGCN(nn.Module):
    """Undirected Line-Graph / Conflated Isotropic L1 Edge GCN Baseline.

    Conflates lower and upper adjacencies without orientation-aware decoupled
    ``L1_down`` and ``L1_up`` filter banks (mixes unsigned line-graph diffusion with
    a single tied ``L1`` weight).
    """

    def __init__(
        self,
        sc: SimplicialComplex,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 1,
        num_layers: int = 4,
        equivariant: bool = True,
        edge_orientations: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        ops_f64 = build_torch_operators(
            sc, dtype=torch.float64, edge_orientations=edge_orientations
        )
        # Unsigned undirected line-graph operator conflating edge orientations & simplices
        conflated_f64 = 0.75 * ops_f64["LG_norm"] + 0.25 * ops_f64["L1_iso_norm"]
        self._op_conflated_f64 = conflated_f64
        self.register_buffer("op_conflated", conflated_f64.to(dtype=dtype))

        use_bias = not equivariant
        self.act = torch.tanh if equivariant else nn.GELU()

        self.in_proj = nn.Linear(in_channels, hidden_channels, bias=use_bias)
        self.self_lins = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels, bias=use_bias) for _ in range(num_layers)]
        )
        self.neigh_lins = nn.ModuleList(
            [nn.Linear(hidden_channels, hidden_channels, bias=use_bias) for _ in range(num_layers)]
        )
        self.out_proj = nn.Linear(hidden_channels, out_channels, bias=use_bias)
        if dtype != torch.float32:
            self.to(dtype=dtype)

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "_op_conflated_f64") and self.op_conflated.dtype == torch.float64:
            self.op_conflated.copy_(self._op_conflated_f64.to(device=self.op_conflated.device))
        return self

    def encode_edges(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.act(self.in_proj(x))
        for lin_s, lin_n in zip(self.self_lins, self.neigh_lins):
            nh = _op_mul(self.op_conflated, h)
            h = h + self.act(lin_s(h) + lin_n(nh))
        return h

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_3d, _, orig_mode = _format_edge_input(x, mask)
        h = self.encode_edges(x_3d, mask=mask)
        out = self.out_proj(h)
        return _restore_edge_output(out, orig_mode)


class EdgeMLP(nn.Module):
    """Topology-agnostic per-edge MLP baseline."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 1,
        num_layers: int = 4,
        equivariant: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        use_bias = not equivariant
        self.act = torch.tanh if equivariant else nn.GELU()

        layers: List[nn.Module] = [nn.Linear(in_channels, hidden_channels, bias=use_bias)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_channels, hidden_channels, bias=use_bias))
        self.layers = nn.ModuleList(layers)
        self.out_proj = nn.Linear(hidden_channels, out_channels, bias=use_bias)
        if dtype != torch.float32:
            self.to(dtype=dtype)

    def encode_edges(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = self.act(layer(h))
        return h

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_3d, _, orig_mode = _format_edge_input(x, mask)
        h = self.encode_edges(x_3d, mask=mask)
        out = self.out_proj(h)
        return _restore_edge_output(out, orig_mode)


class SimplicialTrajectoryClassifier(nn.Module):
    """Gauge-invariant topological trajectory classifier with unified readout across backbones.

    Pools latent 1-cochains via strictly ``D = diag(+-1)`` orientation-invariant L1/L2
    norms and inner-boundary 1-cycle duality pairings ``C_bnd.T @ h``, applying the
    exact same readout architecture and 4-layer depth across all four backbones.
    """

    def __init__(
        self,
        sc: SimplicialComplex,
        backbone_type: str = "hodge_snn",
        hidden_channels: int = 32,
        num_classes: int = 4,
        edge_orientations: Optional[np.ndarray] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.backbone_type = backbone_type
        ops_f64 = build_torch_operators(
            sc, dtype=torch.float64, edge_orientations=edge_orientations
        )
        self._c_bnd_f64 = ops_f64["C_bnd"]
        self.register_buffer("C_bnd", ops_f64["C_bnd"].to(dtype=dtype))

        if backbone_type == "hodge_snn":
            self.backbone = HodgeSimplicialNN(
                sc,
                in_channels=1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=4,
                filter_order=2,
                equivariant=True,
                edge_orientations=edge_orientations,
                dtype=dtype,
            )
            feat_dim = hidden_channels * 2
        elif backbone_type == "node_gcn":
            self.backbone = NodeGCNEdgePredictor(
                sc,
                in_channels=1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=4,
                equivariant=True,
                edge_orientations=edge_orientations,
                dtype=dtype,
            )
            feat_dim = hidden_channels
        elif backbone_type == "isotropic_l1":
            self.backbone = EdgeLineGraphGCN(
                sc,
                in_channels=1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=4,
                equivariant=True,
                edge_orientations=edge_orientations,
                dtype=dtype,
            )
            feat_dim = hidden_channels
        elif backbone_type == "edge_mlp":
            self.backbone = EdgeMLP(
                in_channels=1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=4,
                equivariant=True,
                dtype=dtype,
            )
            feat_dim = hidden_channels
        else:
            raise ValueError(f"Unknown backbone_type '{backbone_type}'.")

        n_cycles = int(self.C_bnd.shape[1])
        pool_dim = feat_dim * (2 + n_cycles)

        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_classes),
        )
        if dtype != torch.float32:
            self.to(dtype=dtype)

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "_c_bnd_f64") and self.C_bnd.dtype == torch.float64:
            self.C_bnd.copy_(self._c_bnd_f64.to(device=self.C_bnd.device))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_3d, _, _ = _format_edge_input(x)
        h = self.backbone.encode_edges(x_3d)  # (B, N1, feat_dim)
        l1_pool = torch.abs(h).mean(dim=1)
        rms_pool = torch.sqrt(torch.mean(h ** 2, dim=1) + 1e-8)
        feats = [l1_pool, rms_pool]
        if self.C_bnd.shape[1] > 0:
            # Gauge-invariant 1-chain / 1-cochain boundary pairing: (D C_bnd)^T (D h) == C_bnd^T h
            cycle_proj = _op_mul(self.C_bnd.transpose(0, 1).contiguous(), h)
            feats.append(cycle_proj.reshape(h.shape[0], -1))
        pooled = torch.cat(feats, dim=-1)
        return self.classifier(pooled)

    def with_edge_orientations(
        self,
        sc: SimplicialComplex,
        edge_orientations: np.ndarray,
    ) -> "SimplicialTrajectoryClassifier":
        """Clone learnable parameters onto ``sc`` with gauge-flipped edge orientations ``D = diag(+-1)``."""
        lin0 = self.classifier[0]
        lin1 = self.classifier[2]
        assert isinstance(lin0, nn.Linear) and isinstance(lin1, nn.Linear)
        clone = SimplicialTrajectoryClassifier(
            sc=sc,
            backbone_type=self.backbone_type,
            hidden_channels=lin0.out_features,
            num_classes=lin1.out_features,
            edge_orientations=edge_orientations,
            dtype=lin0.weight.dtype,
        ).to(device=lin0.weight.device)
        src_params = dict(self.named_parameters())
        with torch.no_grad():
            for name, p_dst in clone.named_parameters():
                p_dst.copy_(src_params[name])
        if not self.training:
            clone.eval()
        return clone


# ---------------------------------------------------------------------------
# Benchmark Dataset Generators & Evaluators
# ---------------------------------------------------------------------------


def generate_edge_imputation_dataset(
    sc: SimplicialComplex,
    n_samples: int = 200,
    mask_ratio: float = 0.40,
    solenoidal_fraction: float = 0.70,
    noise_std: float = 0.03,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate multi-component physical flows on ``sc`` with masked & noisy observations.

    Controls the energy split between irrotational gradient flow ``(1 - eta_sol)``
    and divergence-free solenoidal (curl + harmonic) flow ``eta_sol``.
    """
    rng = np.random.default_rng(seed)
    n0, n1, n2 = sc.shape_tuple
    b1, b2 = sc.B1, sc.B2
    u_harm = harmonic_basis(sc)
    beta_1 = u_harm.shape[1]

    # Precompute smooth spectral bases of L0 and L1_up so flows are spatially coherent
    l0 = graph_laplacian_0(sc)
    evals0, evecs0 = np.linalg.eigh(l0)
    # Take low-frequency non-constant modes of L0 for smooth scalar potentials
    pos0 = np.where(evals0 > 1e-6)[0][: min(10, max(1, n0 - 1))]
    phi_modes = evecs0[:, pos0]

    if n2 > 0:
        # Smooth 2-simplex dual graph Laplacian modes for spatially coherent vortices
        dual_adj = ( np.abs(b2.T @ b2 - np.diag(np.diag(b2.T @ b2))) > 0 ).astype(np.float64)
        dual_lap = np.diag(dual_adj.sum(axis=1)) - dual_adj
        _, evecs_dual = np.linalg.eigh(dual_lap)
        psi_modes = evecs_dual[:, : min(10, n2)]
    else:
        psi_modes = np.zeros((0, 0), dtype=np.float64)

    eta = float(np.clip(solenoidal_fraction, 0.0, 1.0))
    w_grad = np.sqrt(max(0.0, 1.0 - eta))
    # Split solenoidal energy evenly between curl (if N2>0) and harmonic (if beta_1>0)
    if n2 > 0 and beta_1 > 0:
        w_curl = np.sqrt(0.5 * eta)
        w_harm = np.sqrt(0.5 * eta)
    elif n2 > 0:
        w_curl = np.sqrt(eta)
        w_harm = 0.0
    else:
        w_curl = 0.0
        w_harm = np.sqrt(eta)

    x_obs_all = np.zeros((n_samples, n1), dtype=np.float32)
    y_true_all = np.zeros((n_samples, n1), dtype=np.float32)
    mask_obs_all = np.zeros((n_samples, n1), dtype=np.float32)

    n_masked = max(1, int(round(mask_ratio * n1)))

    for s in range(n_samples):
        c0 = rng.standard_normal(phi_modes.shape[1])
        fg = b1.T @ (phi_modes @ c0)
        fg = fg / max(np.linalg.norm(fg), 1e-12)

        if n2 > 0:
            c2 = rng.standard_normal(psi_modes.shape[1])
            fc = b2 @ (psi_modes @ c2)
            fc = fc / max(np.linalg.norm(fc), 1e-12)
        else:
            fc = np.zeros(n1, dtype=np.float64)

        if beta_1 > 0:
            ch = rng.standard_normal(beta_1)
            if np.linalg.norm(ch) < 1e-6:
                ch[0] = 1.0
            fh = u_harm @ ch
            fh = fh / max(np.linalg.norm(fh), 1e-12)
        else:
            fh = np.zeros(n1, dtype=np.float64)

        amp = np.sqrt(float(n1))
        f_true = amp * (w_grad * fg + w_curl * fc + w_harm * fh)

        obs_mask = np.ones(n1, dtype=np.float64)
        masked_idx = rng.choice(n1, size=n_masked, replace=False)
        obs_mask[masked_idx] = 0.0

        noisy = f_true + noise_std * np.std(f_true) * rng.standard_normal(n1)
        x_obs = obs_mask * noisy

        x_obs_all[s] = x_obs.astype(np.float32)
        y_true_all[s] = f_true.astype(np.float32)
        mask_obs_all[s] = obs_mask.astype(np.float32)

    return {
        "x_obs": torch.from_numpy(x_obs_all).unsqueeze(-1),
        "y_true": torch.from_numpy(y_true_all).unsqueeze(-1),
        "mask_obs": torch.from_numpy(mask_obs_all).unsqueeze(-1),
    }


def compute_r2_and_mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eval_mask: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Compute R^2 determination coefficient and MSE on target entries (e.g. missing edges)."""
    yp = y_pred.detach().cpu().numpy().reshape(-1)
    yt = y_true.detach().cpu().numpy().reshape(-1)
    if eval_mask is not None:
        m = eval_mask.detach().cpu().numpy().reshape(-1) > 0.5
        yp = yp[m]
        yt = yt[m]
    mse = float(np.mean((yp - yt) ** 2))
    var = float(np.var(yt))
    r2 = 1.0 - (mse / max(var, 1e-12))
    return float(r2), mse


def train_and_eval_imputation(
    sc: SimplicialComplex,
    solenoidal_fraction: float = 0.70,
    mask_ratio: float = 0.40,
    n_train: int = 160,
    n_test: int = 60,
    epochs: int = 140,
    hidden_channels: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Train all 4 architectures on Missing Edge Flux Imputation & evaluate on held-out flows."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = generate_edge_imputation_dataset(
        sc,
        n_samples=n_train,
        mask_ratio=mask_ratio,
        solenoidal_fraction=solenoidal_fraction,
        seed=seed,
    )
    test_data = generate_edge_imputation_dataset(
        sc,
        n_samples=n_test,
        mask_ratio=mask_ratio,
        solenoidal_fraction=solenoidal_fraction,
        seed=seed + 10_000,
    )

    models: Dict[str, nn.Module] = {
        "NodeGCN": NodeGCNEdgePredictor(
            sc, in_channels=1, hidden_channels=hidden_channels, out_channels=1, equivariant=True
        ),
        "EdgeMLP": EdgeMLP(
            in_channels=1, hidden_channels=hidden_channels, out_channels=1, equivariant=True
        ),
        "Isotropic-L1-GCN": EdgeLineGraphGCN(
            sc, in_channels=1, hidden_channels=hidden_channels, out_channels=1, equivariant=True
        ),
        "HodgeSimplicialNN": HodgeSimplicialNN(
            sc, in_channels=1, hidden_channels=hidden_channels, out_channels=1, equivariant=True
        ),
    }

    x_tr, y_tr, m_tr = train_data["x_obs"], train_data["y_true"], train_data["mask_obs"]
    x_te, y_te, m_te = test_data["x_obs"], test_data["y_true"], test_data["mask_obs"]
    missing_tr = 1.0 - m_tr
    missing_te = 1.0 - m_te

    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(x_tr, mask=m_tr)
            # Loss weighted on missing edges (80%) + denoising observed edges (20%)
            loss_miss = ((pred - y_tr) ** 2 * missing_tr).sum() / missing_tr.sum().clamp_min(1.0)
            loss_obs = ((pred - y_tr) ** 2 * m_tr).sum() / m_tr.sum().clamp_min(1.0)
            loss = 0.85 * loss_miss + 0.15 * loss_obs
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_te = model(x_te, mask=m_te)
            r2_miss, mse_miss = compute_r2_and_mse(pred_te, y_te, eval_mask=missing_te)
            r2_all, mse_all = compute_r2_and_mse(pred_te, y_te, eval_mask=None)
        results[name] = {
            "missing_r2": r2_miss,
            "missing_mse": mse_miss,
            "overall_r2": r2_all,
            "overall_mse": mse_all,
        }
    return results


# ---------------------------------------------------------------------------
# Task 2: Topological Homotopy Trajectory Classification
# ---------------------------------------------------------------------------


def _shortest_edge_walk(
    adj_list: Dict[int, List[Tuple[int, float]]],
    start: int,
    goal: int,
) -> List[int]:
    """Dijkstra shortest path between two vertices on weighted 1-skeleton."""
    pq: List[Tuple[float, int]] = [(0.0, start)]
    dist: Dict[int, float] = {start: 0.0}
    parent: Dict[int, int] = {}
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj_list.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    if goal != start and goal not in parent:
        return [start]
    path = [goal]
    curr = goal
    while curr != start:
        curr = parent[curr]
        path.append(curr)
    path.reverse()
    return path


def generate_homotopy_trajectory_dataset(
    sc: SimplicialComplex,
    n_per_class: int = 50,
    radius_range: Tuple[float, float] = (0.30, 0.55),
    curl_noise_std: float = 0.45,
    grad_noise_std: float = 0.25,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate discrete oriented trajectory 1-cochains in 4 homotopy classes.

    Classes on ``planar_punctured_complex``:
    - Class 0: Counter-clockwise loop winding around Hole A ``(+1, 0)``
    - Class 1: Clockwise loop winding around Hole A ``(-1, 0)``
    - Class 2: Counter-clockwise loop winding around Hole B ``(0, +1)``
    - Class 3: Contractible loop in filled 2-simplex region ``(0, 0)``

    Adds both triangle-hopping homotopy deformations (``B2 @ psi_noise``) and
    local backtrack gradient spurs (``B1.T @ phi_noise``).
    """
    if sc.coords is None:
        raise ValueError("generate_homotopy_trajectory_dataset requires 2D coords on sc.")
    rng = np.random.default_rng(seed)
    coords = sc.coords[:, :2]
    n0, n1, n2 = sc.shape_tuple
    b1, b2 = sc.B1, sc.B2

    adj_base: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n0)}
    for u, v in sc.edges:
        iu, iv = int(u), int(v)
        elen = float(np.linalg.norm(coords[iv] - coords[iu]))
        adj_base[iu].append((iv, elen))
        adj_base[iv].append((iu, elen))

    centers = sc.metadata.get("hole_centers", [(-0.44, 0.0), (0.44, 0.0)])
    hole_a = np.array(centers[0], dtype=np.float64)
    hole_b = np.array(centers[1], dtype=np.float64)
    # Central filled bridge / upper-lower contractible loop centers
    contractible_centers = [
        np.array([0.0, 0.22], dtype=np.float64),
        np.array([0.0, -0.22], dtype=np.float64),
    ]

    total = 4 * n_per_class
    x_list = np.zeros((total, n1), dtype=np.float32)
    y_list = np.zeros(total, dtype=np.int64)

    idx_out = 0
    for cls_id in range(4):
        for _ in range(n_per_class):
            rad = rng.uniform(radius_range[0], radius_range[1])
            phase0 = rng.uniform(0.0, 2.0 * np.pi)
            n_wp = 6

            if cls_id == 0:
                center = hole_a
                angles = phase0 + np.linspace(0.0, 2.0 * np.pi, n_wp, endpoint=False)
            elif cls_id == 1:
                center = hole_a
                angles = phase0 - np.linspace(0.0, 2.0 * np.pi, n_wp, endpoint=False)
            elif cls_id == 2:
                center = hole_b
                angles = phase0 + np.linspace(0.0, 2.0 * np.pi, n_wp, endpoint=False)
            else:
                center = contractible_centers[int(rng.integers(0, len(contractible_centers)))]
                rad_c = rng.uniform(0.14, 0.24)
                sign_c = 1.0 if rng.random() < 0.5 else -1.0
                angles = phase0 + sign_c * np.linspace(0.0, 2.0 * np.pi, n_wp, endpoint=False)
                rad = rad_c

            # Random edge-weight jitter so shortest-path segments vary stochastically
            adj_jit: Dict[int, List[Tuple[int, float]]] = {
                u: [(v, w * float(rng.uniform(0.8, 1.35))) for v, w in nbrs]
                for u, nbrs in adj_base.items()
            }

            waypoints: List[int] = []
            for ang in angles:
                target_pt = center + np.array([rad * np.cos(ang), 0.85 * rad * np.sin(ang)])
                dists = np.linalg.norm(coords - target_pt[None, :], axis=1)
                wp = int(np.argmin(dists))
                if not waypoints or waypoints[-1] != wp:
                    waypoints.append(wp)

            cochain = np.zeros(n1, dtype=np.float64)
            if len(waypoints) >= 2:
                cyc_wps = waypoints + [waypoints[0]]
                for a_wp, b_wp in zip(cyc_wps[:-1], cyc_wps[1:]):
                    walk = _shortest_edge_walk(adj_jit, a_wp, b_wp)
                    for u_step, v_step in zip(walk[:-1], walk[1:]):
                        e_key = _canonical_edge(u_step, v_step)
                        e_idx = sc.edge_to_idx[e_key]
                        sign = 1.0 if u_step < v_step else -1.0
                        cochain[e_idx] += sign

            # Homotopy-preserving curl deformations across filled 2-simplices + gradient spurs
            if n2 > 0 and curl_noise_std > 0:
                psi_pert = curl_noise_std * rng.standard_normal(n2)
                cochain = cochain + (b2 @ psi_pert)
            if grad_noise_std > 0:
                phi_pert = grad_noise_std * rng.standard_normal(n0)
                cochain = cochain + (b1.T @ phi_pert)

            x_list[idx_out] = cochain.astype(np.float32)
            y_list[idx_out] = cls_id
            idx_out += 1

    perm = rng.permutation(total)
    return {
        "x": torch.from_numpy(x_list[perm]).unsqueeze(-1),
        "y": torch.from_numpy(y_list[perm]),
    }


def train_and_eval_homotopy_classification(
    sc: SimplicialComplex,
    n_train_per_class: int = 45,
    n_test_per_class: int = 30,
    epochs: int = 120,
    hidden_channels: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
) -> Dict[str, float]:
    """Train & evaluate all 4 architectures on unseen radial homotopy trajectories."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train on tighter inner loops, evaluate generalization on wider outer loops!
    train_ds = generate_homotopy_trajectory_dataset(
        sc,
        n_per_class=n_train_per_class,
        radius_range=(0.27, 0.39),
        curl_noise_std=0.45,
        grad_noise_std=0.25,
        seed=seed,
    )
    test_ds = generate_homotopy_trajectory_dataset(
        sc,
        n_per_class=n_test_per_class,
        radius_range=(0.43, 0.62),
        curl_noise_std=0.55,
        grad_noise_std=0.30,
        seed=seed + 5000,
    )

    specs = [
        ("NodeGCN", "node_gcn"),
        ("EdgeMLP", "edge_mlp"),
        ("Isotropic-L1-GCN", "isotropic_l1"),
        ("HodgeSimplicialNN", "hodge_snn"),
    ]
    criterion = nn.CrossEntropyLoss()
    accuracies: Dict[str, float] = {}

    x_tr, y_tr = train_ds["x"], train_ds["y"]
    x_te, y_te = test_ds["x"], test_ds["y"]

    # The four generated loop classes have signed winding around the two cavities.
    # Fit a geometric control on the same training samples and score the held-out radii.
    basis = harmonic_basis(sc)
    train_coords = x_tr.squeeze(-1).numpy() @ basis
    test_coords = x_te.squeeze(-1).numpy() @ basis
    centers = np.stack([train_coords[y_tr.numpy() == c].mean(axis=0) for c in range(4)])
    distances = np.linalg.norm(test_coords[:, None, :] - centers[None, :, :], axis=-1)
    accuracies["HarmonicCentroid"] = float(np.mean(distances.argmin(axis=1) == y_te.numpy()))

    for label, btype in specs:
        torch.manual_seed(seed)
        clf = SimplicialTrajectoryClassifier(
            sc, backbone_type=btype, hidden_channels=hidden_channels, num_classes=4
        )
        opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
        clf.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = clf(x_tr)
            loss = criterion(logits, y_tr)
            loss.backward()
            opt.step()

        clf.eval()
        with torch.no_grad():
            preds = clf(x_te).argmax(dim=-1)
            acc = float((preds == y_te).float().mean().item())
        accuracies[label] = acc

    return accuracies
