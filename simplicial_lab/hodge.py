"""Hodge-Laplacian Spectral Topology & Exact Helmholtz-Hodge Decomposition.

Implements discrete exterior calculus operators on 2-simplicial complexes:
- 0-Laplacian L0 = B1 @ B1.T
- 1-Hodge Laplacian L1 = L1_down + L1_up = B1.T @ B1 + B2 @ B2.T
- Exact Betti numbers (beta_0, beta_1) via SVD rank-nullity and Hodge spectra
- Orthonormal harmonic 1-cochain basis spanning ker(L1)
- Exact orthogonal Helmholtz-Hodge Decomposition f = f_grad + f_curl + f_harm
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .complexes import SimplicialComplex


def graph_laplacian_0(sc: SimplicialComplex) -> np.ndarray:
    """Compute the 0-Hodge (combinatorial graph) Laplacian ``L0 = B1 @ B1.T``."""
    b1 = sc.B1
    return b1 @ b1.T


def hodge_laplacian_1(
    sc: SimplicialComplex,
    return_components: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute the 1-Hodge Laplacian ``L1 = L1_down + L1_up = B1.T @ B1 + B2 @ B2.T``.

    Parameters
    ----------
    sc : SimplicialComplex
        Oriented 2-simplicial complex.
    return_components : bool
        If True, returns ``(L1, L1_down, L1_up)``. Otherwise returns ``L1``.
    """
    b1 = sc.B1
    b2 = sc.B2
    l1_down = b1.T @ b1
    if sc.num_triangles > 0:
        l1_up = b2 @ b2.T
    else:
        l1_up = np.zeros_like(l1_down)
    l1 = l1_down + l1_up
    if return_components:
        return l1, l1_down, l1_up
    return l1


def compute_betti_numbers(
    sc: SimplicialComplex,
    tol: float = 1e-8,
    method: str = "svd",
) -> Tuple[int, int]:
    """Compute exact topological Betti numbers ``(beta_0, beta_1)``.

    Parameters
    ----------
    sc : SimplicialComplex
        Target simplicial complex.
    tol : float
        Numerical rank / zero-eigenvalue threshold.
    method : {"svd", "spectral"}
        ``"svd"`` uses rank-nullity on ``B1`` and ``B2``; ``"spectral"`` counts
        zero eigenvalues of ``L0`` and ``L1``.
    """
    if method == "svd":
        s1 = np.linalg.svd(sc.B1, compute_uv=False) if sc.num_edges > 0 else np.array([])
        tol_s1 = tol * max(1.0, float(s1[0])) if len(s1) > 0 else tol
        rank_b1 = int(np.sum(s1 > tol_s1))
        s2 = (
            np.linalg.svd(sc.B2, compute_uv=False)
            if (sc.num_triangles > 0 and sc.num_edges > 0)
            else np.array([])
        )
        tol_s2 = tol * max(1.0, float(s2[0])) if len(s2) > 0 else tol
        rank_b2 = int(np.sum(s2 > tol_s2))
        beta_0 = int(sc.num_nodes - rank_b1)
        beta_1 = int(sc.num_edges - rank_b1 - rank_b2)
        return beta_0, beta_1
    elif method == "spectral":
        l0 = graph_laplacian_0(sc)
        evals0 = np.linalg.eigvalsh(l0)
        tol_l0 = tol * max(1.0, float(np.max(np.abs(evals0)))) if len(evals0) > 0 else tol
        beta_0 = int(np.sum(np.abs(evals0) <= tol_l0))
        if sc.num_edges == 0:
            return beta_0, 0
        l1 = hodge_laplacian_1(sc)
        evals1 = np.linalg.eigvalsh(l1)
        tol_l1 = tol * max(1.0, float(np.max(np.abs(evals1)))) if len(evals1) > 0 else tol
        beta_1 = int(np.sum(np.abs(evals1) <= tol_l1))
        return beta_0, beta_1
    else:
        raise ValueError(f"Unknown Betti computation method '{method}'. Choose 'svd' or 'spectral'.")


def edge_winding_operator(
    sc: SimplicialComplex,
    centers: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Compute discrete angular 1-forms ``d(theta_c)`` on edges around reference points.

    Returns matrix ``W`` of shape ``(N1, len(centers))`` where ``W[e, m]`` is the
    signed angular change ``Delta theta`` from ``i -> j`` around center ``m`` in ``(-pi, pi]``.
    """
    if sc.coords is None:
        raise ValueError("edge_winding_operator requires geometric coordinates on sc.coords.")
    if centers is None:
        centers = sc.metadata.get("hole_centers", [(0.0, 0.0)])
    n_edges = sc.num_edges
    w_mat = np.zeros((n_edges, len(centers)), dtype=np.float64)
    if n_edges == 0 or len(centers) == 0:
        return w_mat

    p_i = sc.coords[sc.edges[:, 0], :2]
    p_j = sc.coords[sc.edges[:, 1], :2]
    for m, (cx, cy) in enumerate(centers):
        c = np.array([cx, cy], dtype=np.float64)
        ri = p_i - c
        rj = p_j - c
        th_i = np.arctan2(ri[:, 1], ri[:, 0])
        th_j = np.arctan2(rj[:, 1], rj[:, 0])
        dth = (th_j - th_i + np.pi) % (2.0 * np.pi) - np.pi
        w_mat[:, m] = dth / (2.0 * np.pi)
    return w_mat


def harmonic_basis(
    sc: SimplicialComplex,
    tol: float = 1e-8,
    align_to_holes: bool = True,
    edge_orientations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return an orthonormal basis ``U_harm`` of shape ``(N1, beta_1)`` spanning ``ker(L1)``."""
    if sc.num_edges == 0:
        return np.zeros((0, 0), dtype=np.float64)

    signs: Optional[np.ndarray] = None
    if edge_orientations is not None:
        signs = np.asarray(edge_orientations, dtype=np.float64).reshape(-1)
        if signs.shape[0] != sc.num_edges or not np.all(np.isin(np.abs(signs), [1.0])):
            raise ValueError("edge_orientations must have shape (N1,) with entries in {-1, +1}.")
        b1_s = sc.B1 * signs[None, :]
        b2_s = signs[:, None] * sc.B2 if sc.num_triangles > 0 else sc.B2
        l1 = b1_s.T @ b1_s + (b2_s @ b2_s.T if sc.num_triangles > 0 else 0.0)
    else:
        l1 = hodge_laplacian_1(sc)

    evals, evecs = np.linalg.eigh(l1)
    tol_eff = tol * max(1.0, float(np.max(np.abs(evals)))) if len(evals) > 0 else tol
    mask = np.abs(evals) <= tol_eff
    u_harm = evecs[:, mask]
    beta_1 = u_harm.shape[1]

    if (
        align_to_holes
        and beta_1 >= 2
        and sc.coords is not None
        and "hole_centers" in sc.metadata
    ):
        centers = sc.metadata["hole_centers"]
        if len(centers) == beta_1:
            w_forms = edge_winding_operator(sc, centers)
            if signs is not None:
                w_forms = signs[:, None] * w_forms
            # Orthogonal Procrustes alignment inside ker(L1) so U_harm remains strictly orthonormal
            coupling = u_harm.T @ w_forms
            u_svd, _, vt_svd = np.linalg.svd(coupling)
            q_orth = u_svd @ vt_svd
            u_harm = u_harm @ q_orth
            # Ensure positive CCW orientation around each hole
            diag_signs = np.sign(np.diag(u_harm.T @ w_forms))
            diag_signs[diag_signs == 0.0] = 1.0
            u_harm = u_harm * diag_signs[None, :]
    return u_harm


@dataclass
class HHDResult:
    """Container for orthogonal Helmholtz-Hodge Decomposition of a 1-cochain."""

    f_total: np.ndarray
    f_grad: np.ndarray
    f_curl: np.ndarray
    f_harm: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    energies: Dict[str, float]
    energy_fractions: Dict[str, float]
    orthogonality_errors: Dict[str, float]
    parseval_error: float

    def __iter__(self) -> Iterator[np.ndarray]:
        """Allow tuple unpacking ``f_grad, f_curl, f_harm = helmholtz_hodge_decomposition(sc, f)``."""
        yield self.f_grad
        yield self.f_curl
        yield self.f_harm


def helmholtz_hodge_decomposition(
    sc: SimplicialComplex,
    f: np.ndarray,
    rcond: float = 1e-12,
) -> HHDResult:
    """Compute exact orthogonal Helmholtz-Hodge Decomposition ``f = f_grad + f_curl + f_harm``.

    Uses minimum-norm SVD least-squares projections onto ``im(B1.T)`` and ``im(B2)``:
    - ``f_grad = B1.T @ phi`` (curl-free / exact 1-coboundary)
    - ``f_curl = B2 @ psi`` (divergence-free / coexact 1-cochain)
    - ``f_harm = f - f_grad - f_curl`` (topological harmonic 1-cochain in ``ker(L1)``)
    """
    f_vec = np.asarray(f, dtype=np.float64).reshape(-1)
    if f_vec.shape[0] != sc.num_edges:
        raise ValueError(
            f"Edge cochain length ({f_vec.shape[0]}) must match sc.num_edges ({sc.num_edges})."
        )

    b1 = sc.B1
    b2 = sc.B2

    if sc.num_edges > 0:
        phi, *_ = np.linalg.lstsq(b1.T, f_vec, rcond=rcond)
        f_grad = b1.T @ phi
    else:
        phi = np.zeros(sc.num_nodes, dtype=np.float64)
        f_grad = np.zeros(0, dtype=np.float64)

    residual = f_vec - f_grad
    if sc.num_triangles > 0 and sc.num_edges > 0:
        psi, *_ = np.linalg.lstsq(b2, residual, rcond=rcond)
        f_curl = b2 @ psi
    else:
        psi = np.zeros(sc.num_triangles, dtype=np.float64)
        f_curl = np.zeros_like(f_vec)

    f_harm = f_vec - f_grad - f_curl

    e_tot = float(np.dot(f_vec, f_vec))
    e_grad = float(np.dot(f_grad, f_grad))
    e_curl = float(np.dot(f_curl, f_curl))
    e_harm = float(np.dot(f_harm, f_harm))
    denom = max(e_tot, 1e-30)

    ip_gc = float(np.abs(np.dot(f_grad, f_curl)))
    ip_gh = float(np.abs(np.dot(f_grad, f_harm)))
    ip_ch = float(np.abs(np.dot(f_curl, f_harm)))
    max_ip = max(ip_gc, ip_gh, ip_ch)
    parseval_err = float(np.abs(e_tot - (e_grad + e_curl + e_harm)))

    return HHDResult(
        f_total=f_vec,
        f_grad=f_grad,
        f_curl=f_curl,
        f_harm=f_harm,
        phi=phi,
        psi=psi,
        energies={
            "total": e_tot,
            "grad": e_grad,
            "curl": e_curl,
            "harm": e_harm,
        },
        energy_fractions={
            "grad": e_grad / denom if e_tot > 0 else 0.0,
            "curl": e_curl / denom if e_tot > 0 else 0.0,
            "harm": e_harm / denom if e_tot > 0 else 0.0,
        },
        orthogonality_errors={
            "grad_curl": ip_gc,
            "grad_harm": ip_gh,
            "curl_harm": ip_ch,
            "max": max_ip,
        },
        parseval_error=parseval_err,
    )


def synthesize_hodge_flow(
    sc: SimplicialComplex,
    grad_weight: float = 1.0,
    curl_weight: float = 1.0,
    harm_weight: float = 1.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Synthesize a ground-truth edge flow ``f = f_grad + f_curl + f_harm`` with controlled norms.

    Constructs smooth geometric potentials when ``sc.coords`` is available, or random
    spectral potentials otherwise, normalizing each active component before weighting.
    """
    rng = np.random.default_rng(seed)
    n0, n1, n2 = sc.shape_tuple
    b1, b2 = sc.B1, sc.B2

    # 1. Gradient flow from smooth source-sink scalar potential phi
    if sc.coords is not None:
        x = sc.coords[:, 0]
        y = sc.coords[:, 1]
        alpha = rng.uniform(0.7, 1.3)
        beta = rng.uniform(-0.5, 0.5)
        src = np.exp(-2.8 * ((x + 0.85) ** 2 + (y - 0.35 * beta) ** 2))
        snk = -np.exp(-2.8 * ((x - 0.85) ** 2 + (y + 0.35 * beta) ** 2))
        phi_raw = alpha * x + 0.25 * beta * y + 0.9 * (src + snk) + 0.06 * rng.standard_normal(n0)
    else:
        phi_raw = rng.standard_normal(n0)

    f_grad_unit = b1.T @ phi_raw
    norm_g = np.linalg.norm(f_grad_unit)
    if norm_g > 1e-12:
        f_grad_unit = f_grad_unit / norm_g
        phi_true = phi_raw / norm_g * grad_weight
    else:
        phi_true = np.zeros(n0, dtype=np.float64)
    f_grad = float(grad_weight) * f_grad_unit

    # 2. Curl flow from localized eddy vorticity psi on 2-simplices
    if n2 > 0:
        if sc.coords is not None:
            tri_centroids = sc.coords[sc.triangles].mean(axis=1)
            cx1, cy1 = 0.0, 0.42
            cx2, cy2 = 0.0, -0.42
            vort1 = np.exp(-9.0 * ((tri_centroids[:, 0] - cx1) ** 2 + (tri_centroids[:, 1] - cy1) ** 2))
            vort2 = -0.9 * np.exp(
                -9.0 * ((tri_centroids[:, 0] - cx2) ** 2 + (tri_centroids[:, 1] - cy2) ** 2)
            )
            psi_raw = vort1 + vort2 + 0.12 * rng.standard_normal(n2)
        else:
            psi_raw = rng.standard_normal(n2)
        f_curl_unit = b2 @ psi_raw
        norm_c = np.linalg.norm(f_curl_unit)
        if norm_c > 1e-12:
            f_curl_unit = f_curl_unit / norm_c
            psi_true = psi_raw / norm_c * curl_weight
        else:
            psi_true = np.zeros(n2, dtype=np.float64)
        f_curl = float(curl_weight) * f_curl_unit
    else:
        psi_true = np.zeros(0, dtype=np.float64)
        f_curl = np.zeros(n1, dtype=np.float64)

    # 3. Harmonic flow circulating around topological cavities
    u_harm = harmonic_basis(sc)
    if u_harm.shape[1] > 0:
        if u_harm.shape[1] == 2:
            # Counter-rotating or distinct circulation strengths around Hole A & Hole B
            signs = np.array([1.0, -0.82], dtype=np.float64)
            jitter = 1.0 + 0.12 * rng.standard_normal(2)
            coeffs = signs * jitter
        else:
            coeffs = rng.standard_normal(u_harm.shape[1])
            if np.linalg.norm(coeffs) < 1e-8:
                coeffs[0] = 1.0
        f_harm_unit = u_harm @ coeffs
        norm_h = np.linalg.norm(f_harm_unit)
        if norm_h > 1e-12:
            f_harm_unit = f_harm_unit / norm_h
        f_harm = float(harm_weight) * f_harm_unit
    else:
        f_harm = np.zeros(n1, dtype=np.float64)

    f_total = f_grad + f_curl + f_harm
    return {
        "f_total": f_total,
        "f_grad": f_grad,
        "f_curl": f_curl,
        "f_harm": f_harm,
        "phi": phi_true,
        "psi": psi_true,
    }


def edge_cochain_to_node_vectors(
    sc: SimplicialComplex,
    f: np.ndarray,
) -> np.ndarray:
    """Interpolate a 1-cochain ``f`` onto 2D node vector arrows for geometric quiver plotting."""
    if sc.coords is None:
        raise ValueError("SimplicialComplex must have 2D coords to compute node vectors.")
    n0 = sc.num_nodes
    vecs = np.zeros((n0, 2), dtype=np.float64)
    counts = np.zeros(n0, dtype=np.float64)

    for e_idx, (i, j) in enumerate(sc.edges):
        ii, jj = int(i), int(j)
        diff = sc.coords[jj, :2] - sc.coords[ii, :2]
        length = np.linalg.norm(diff)
        if length > 1e-12:
            unit_dir = diff / length
            flow_vec = float(f[e_idx]) * unit_dir
            vecs[ii] += flow_vec
            vecs[jj] += flow_vec
            counts[ii] += 1.0
            counts[jj] += 1.0

    valid = counts > 0
    vecs[valid] /= counts[valid, None]
    return vecs


def cavity_filling_filtration(
    sc: SimplicialComplex,
    num_mesh_steps: int = 25,
    num_lowest_eigs: int = 16,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Track L1 Hodge spectrum and beta_1 as 2-simplices fill faces and seal cavities.

    Phase 1: Progressively adds the complex's mesh 2-simplices from 0% to 100%,
             reducing beta_1 from (N1 - N0 + 1) down to the topological cavity count.
    Phase 2: Sequentially caps each topological hole in ``sc.metadata['hole_cap_triangles']``,
             annihilating the remaining harmonic zero-modes one cavity at a time.
    """
    rng = np.random.default_rng(seed)
    base_tris = [tuple(map(int, t)) for t in sc.triangles]
    perm = rng.permutation(len(base_tris))
    ordered_mesh_tris = [base_tris[i] for i in perm]

    # Sample indices from 0 to len(ordered_mesh_tris)
    counts = np.unique(np.linspace(0, len(ordered_mesh_tris), num_mesh_steps, dtype=int)).tolist()
    stages: List[Tuple[str, List[Tuple[int, int, int]]]] = []
    for c in counts:
        stages.append(("mesh_fill", ordered_mesh_tris[:c]))

    hole_caps: List[List[Tuple[int, int, int]]] = sc.metadata.get("hole_cap_triangles", [])
    accum_tris = list(ordered_mesh_tris)
    for h_idx, cap_tris in enumerate(hole_caps):
        for step_k in range(1, len(cap_tris) + 1):
            stages.append((f"cap_hole_{h_idx + 1}", accum_tris + list(cap_tris[:step_k])))
        accum_tris = accum_tris + list(cap_tris)

    beta1_list: List[int] = []
    gap_list: List[float] = []
    n2_list: List[int] = []
    stage_labels: List[str] = []
    low_eigs = np.zeros((len(stages), num_lowest_eigs), dtype=np.float64)

    for s_idx, (label, tris_now) in enumerate(stages):
        sub_sc = SimplicialComplex(
            num_nodes=sc.num_nodes,
            edges=sc.edges,
            triangles=tris_now,
            coords=sc.coords,
            enforce_closure=True,
        )
        _, b1_val = compute_betti_numbers(sub_sc, tol=1e-8, method="svd")
        l1 = hodge_laplacian_1(sub_sc)
        evals = np.linalg.eigvalsh(l1)
        evals = np.maximum(evals, 0.0)
        k_take = min(num_lowest_eigs, len(evals))
        low_eigs[s_idx, :k_take] = evals[:k_take]

        pos_evals = evals[evals > 1e-7]
        spectral_gap = float(pos_evals[0]) if len(pos_evals) > 0 else 0.0

        beta1_list.append(b1_val)
        gap_list.append(spectral_gap)
        n2_list.append(sub_sc.num_triangles)
        stage_labels.append(label)

    mesh_end_step = len(counts) - 1
    return {
        "step": np.arange(len(stages), dtype=np.int64),
        "num_triangles": np.array(n2_list, dtype=np.int64),
        "beta_1": np.array(beta1_list, dtype=np.int64),
        "spectral_gap": np.array(gap_list, dtype=np.float64),
        "low_eigenvalues": low_eigs,
        "stage_labels": np.array(stage_labels),
        "mesh_complete_step": np.array(mesh_end_step, dtype=np.int64),
    }
