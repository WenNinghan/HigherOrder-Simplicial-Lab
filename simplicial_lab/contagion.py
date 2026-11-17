"""Higher-Order Simplicial Contagion & Synergistic Reaction Dynamics.

Implements the Simplicial Contagion Model (Iacopini et al., Nature Communications 2019):
- Analytical homogeneous mean-field ODE and exact saddle-node bifurcation curves
- Vectorized node-level quenched continuous-time ODE on arbitrary SimplicialComplex
- Microscopic discrete-time Markovian stochastic SIS simulation on SimplicialComplex
- Forward vs backward adiabatic parameter sweeps revealing explosive first-order
  phase transitions and bistable hysteresis loops when lambda_2 > 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .complexes import SimplicialComplex, iacopini_rsc_complex


def analytical_mean_field_equilibria(
    lambda_1: float,
    lambda_2: float,
) -> Dict[str, Optional[float]]:
    """Compute exact steady states of ``d(rho)/d(tau) = -rho + lambda_1*rho*(1-rho) + lambda_2*rho^2*(1-rho)``.

    Returns dictionary with:
    - ``stable_zero``: ``0.0`` if ``lambda_1 <= 1`` else ``None``
    - ``unstable_zero``: ``0.0`` if ``lambda_1 > 1`` else ``None``
    - ``stable_endemic``: upper root ``rho_+^*`` in ``(0, 1]`` if physically valid else ``None``
    - ``unstable_separatrix``: middle root ``rho_-^*`` in ``(0, 1)`` if bistable else ``None``
    """
    l1 = float(lambda_1)
    l2 = float(lambda_2)

    stable_zero: Optional[float] = 0.0 if l1 <= 1.0 else None
    unstable_zero: Optional[float] = 0.0 if l1 > 1.0 else None
    stable_endemic: Optional[float] = None
    unstable_sep: Optional[float] = None

    if abs(l2) < 1e-12:
        if l1 > 1.0:
            stable_endemic = (l1 - 1.0) / l1
    else:
        disc = (l1 + l2) ** 2 - 4.0 * l2
        if disc >= 0.0:
            sqrt_d = float(np.sqrt(disc))
            rho_plus = ((l2 - l1) + sqrt_d) / (2.0 * l2)
            rho_minus = ((l2 - l1) - sqrt_d) / (2.0 * l2)
            if 0.0 < rho_plus <= 1.0:
                stable_endemic = float(rho_plus)
            if 0.0 < rho_minus < 1.0 and disc > 1e-14 and l1 < 1.0 and l2 > 1.0:
                unstable_sep = float(rho_minus)

    return {
        "stable_zero": stable_zero,
        "unstable_zero": unstable_zero,
        "stable_endemic": stable_endemic,
        "unstable_separatrix": unstable_sep,
    }


def analytical_bifurcation_curve(
    lambda_1_grid: np.ndarray,
    lambda_2: float,
) -> Dict[str, np.ndarray]:
    """Evaluate stable and unstable analytical mean-field branches across ``lambda_1_grid``."""
    l1_arr = np.asarray(lambda_1_grid, dtype=np.float64)
    n = len(l1_arr)
    upper_stable = np.full(n, np.nan, dtype=np.float64)
    middle_unstable = np.full(n, np.nan, dtype=np.float64)
    zero_stable = np.full(n, np.nan, dtype=np.float64)
    zero_unstable = np.full(n, np.nan, dtype=np.float64)

    for idx, l1 in enumerate(l1_arr):
        eq = analytical_mean_field_equilibria(float(l1), lambda_2)
        if eq["stable_endemic"] is not None:
            upper_stable[idx] = eq["stable_endemic"]
        if eq["unstable_separatrix"] is not None:
            middle_unstable[idx] = eq["unstable_separatrix"]
        if eq["stable_zero"] is not None:
            zero_stable[idx] = 0.0
        if eq["unstable_zero"] is not None:
            zero_unstable[idx] = 0.0

    l1_tip_back = (
        float(2.0 * np.sqrt(lambda_2) - lambda_2) if lambda_2 > 1.0 else 1.0
    )
    return {
        "lambda_1": l1_arr,
        "upper_stable": upper_stable,
        "middle_unstable": middle_unstable,
        "zero_stable": zero_stable,
        "zero_unstable": zero_unstable,
        "lambda_1_critical_forward": np.array(1.0),
        "lambda_1_critical_backward": np.array(l1_tip_back),
    }


def simulate_mean_field_ode(
    lambda_1: float,
    lambda_2: float,
    rho_init: float = 0.05,
    mu: float = 1.0,
    t_max: float = 60.0,
    dt: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the homogeneous mean-field contagion ODE via 4th-order Runge-Kutta."""
    n_steps = int(np.ceil(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    rho = np.zeros(n_steps + 1, dtype=np.float64)
    rho[0] = float(np.clip(rho_init, 0.0, 1.0))

    def _rhs(r: float) -> float:
        return mu * (-r + lambda_1 * r * (1.0 - r) + lambda_2 * (r ** 2) * (1.0 - r))

    curr = rho[0]
    for s in range(n_steps):
        k1 = _rhs(curr)
        k2 = _rhs(curr + 0.5 * dt * k1)
        k3 = _rhs(curr + 0.5 * dt * k2)
        k4 = _rhs(curr + dt * k3)
        curr = float(np.clip(curr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0))
        rho[s + 1] = curr

    return times, rho


def _prepare_incidence_structures(
    sc: SimplicialComplex,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Build fast vectorized adjacency and 2-simplex incidence arrays for contagion simulation."""
    n0 = sc.num_nodes
    adj = np.zeros((n0, n0), dtype=np.float64)
    if sc.num_edges > 0:
        adj[sc.edges[:, 0], sc.edges[:, 1]] = 1.0
        adj[sc.edges[:, 1], sc.edges[:, 0]] = 1.0

    if sc.num_triangles > 0:
        t = sc.triangles
        # Each triangle (i, j, k) contributes (j, k) to i, (i, k) to j, and (i, j) to k
        tri_directed = np.vstack(
            [
                np.column_stack([t[:, 0], t[:, 1], t[:, 2]]),
                np.column_stack([t[:, 1], t[:, 0], t[:, 2]]),
                np.column_stack([t[:, 2], t[:, 0], t[:, 1]]),
            ]
        ).astype(np.int64)
    else:
        tri_directed = np.empty((0, 3), dtype=np.int64)

    k1_mean, k2_mean = sc.mean_degrees()
    return adj, tri_directed, k1_mean, k2_mean


def simulate_simplicial_contagion_ode(
    sc: SimplicialComplex,
    lambda_1: float,
    lambda_2: float,
    rho_init: float = 0.05,
    mu: float = 1.0,
    t_max: float = 70.0,
    dt: float = 0.1,
    p_init: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Integrate the node-level continuous-time quenched mean-field ODE on a SimplicialComplex.

    Evolves ``p_i(t) in [0, 1]`` for all nodes ``i = 0 .. N0-1`` with microscopic rates
    ``beta_1 = mu * lambda_1 / <k1>`` and ``beta_2 = mu * lambda_2 / <k2>``.
    """
    adj, tri_dir, k1_mean, k2_mean = _prepare_incidence_structures(sc)
    n0 = sc.num_nodes
    beta_1 = (mu * float(lambda_1) / k1_mean) if k1_mean > 0 else 0.0
    beta_2 = (mu * float(lambda_2) / k2_mean) if k2_mean > 0 else 0.0

    if p_init is not None:
        p = np.clip(np.asarray(p_init, dtype=np.float64).copy(), 0.0, 1.0)
    else:
        rng = np.random.default_rng(seed)
        jitter = rng.uniform(0.85, 1.15, size=n0)
        p = np.clip(float(rho_init) * jitter, 0.0, 1.0)
        if np.mean(p) > 0:
            p = np.clip(p * (float(rho_init) / np.mean(p)), 0.0, 1.0)

    n_steps = int(np.ceil(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    rho_traj = np.zeros(n_steps + 1, dtype=np.float64)
    rho_traj[0] = float(np.mean(p))

    has_tris = tri_dir.shape[0] > 0
    if has_tris:
        target_nodes = tri_dir[:, 0]
        src_a = tri_dir[:, 1]
        src_b = tri_dir[:, 2]

    def _rhs(p_vec: np.ndarray) -> np.ndarray:
        inf_1 = adj @ p_vec
        if has_tris:
            prod_ab = p_vec[src_a] * p_vec[src_b]
            inf_2 = np.bincount(target_nodes, weights=prod_ab, minlength=n0)
        else:
            inf_2 = 0.0
        force = beta_1 * inf_1 + beta_2 * inf_2
        return -mu * p_vec + (1.0 - p_vec) * force

    for s in range(n_steps):
        k1 = _rhs(p)
        k2 = _rhs(np.clip(p + 0.5 * dt * k1, 0.0, 1.0))
        k3 = _rhs(np.clip(p + 0.5 * dt * k2, 0.0, 1.0))
        k4 = _rhs(np.clip(p + dt * k3, 0.0, 1.0))
        p = np.clip(p + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0)
        rho_traj[s + 1] = float(np.mean(p))

    return {
        "times": times,
        "rho": rho_traj,
        "p_final": p,
        "rho_steady": float(np.mean(rho_traj[-max(1, n_steps // 10) :])),
    }


def simulate_stochastic_contagion(
    sc: SimplicialComplex,
    lambda_1: float,
    lambda_2: float,
    rho_init: float = 0.10,
    mu: float = 1.0,
    t_max: float = 60.0,
    dt: float = 0.05,
    spontaneous_rate: float = 5e-4,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Run microscopic Markovian discrete-time SIS contagion on a SimplicialComplex."""
    rng = np.random.default_rng(seed)
    adj, tri_dir, k1_mean, k2_mean = _prepare_incidence_structures(sc)
    n0 = sc.num_nodes

    beta_1 = (mu * float(lambda_1) / k1_mean) if k1_mean > 0 else 0.0
    beta_2 = (mu * float(lambda_2) / k2_mean) if k2_mean > 0 else 0.0

    p1_step = min(1.0, beta_1 * dt)
    p2_step = min(1.0, beta_2 * dt)
    p_rec = min(1.0, mu * dt)
    p_spont = min(1.0, spontaneous_rate * dt)

    n_init = max(1, int(round(rho_init * n0))) if rho_init > 0 else 0
    state = np.zeros(n0, dtype=np.float64)
    if n_init > 0:
        chosen = rng.choice(n0, size=min(n0, n_init), replace=False)
        state[chosen] = 1.0

    n_steps = int(np.ceil(t_max / dt))
    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    rho_traj = np.zeros(n_steps + 1, dtype=np.float64)
    rho_traj[0] = float(np.mean(state))

    has_tris = tri_dir.shape[0] > 0
    if has_tris:
        target_nodes = tri_dir[:, 0]
        src_a = tri_dir[:, 1]
        src_b = tri_dir[:, 2]

    for s in range(n_steps):
        n_inf_1 = adj @ state
        if has_tris:
            both_inf = state[src_a] * state[src_b]
            n_inf_2 = np.bincount(target_nodes, weights=both_inf, minlength=n0)
        else:
            n_inf_2 = np.zeros(n0, dtype=np.float64)

        prob_not_inf = ((1.0 - p1_step) ** n_inf_1) * ((1.0 - p2_step) ** n_inf_2) * (1.0 - p_spont)
        prob_inf = 1.0 - prob_not_inf

        u = rng.random(n0)
        new_inf = (state == 0.0) & (u < prob_inf)
        recov = (state == 1.0) & (u < p_rec)

        state[new_inf] = 1.0
        state[recov] = 0.0
        rho_traj[s + 1] = float(np.mean(state))

    steady_window = max(10, n_steps // 4)
    return {
        "times": times,
        "rho": rho_traj,
        "state_final": state,
        "rho_steady": float(np.mean(rho_traj[-steady_window:])),
    }


@dataclass
class HysteresisSweepResult:
    """Results of forward and backward infectivity sweeps on a SimplicialComplex."""

    lambda_1_grid: np.ndarray
    lambda_2: float
    rho_forward: np.ndarray
    rho_backward: np.ndarray
    hysteresis_gap_curve: np.ndarray
    max_hysteresis_gap: float
    bistable_lambda_1_at_max_gap: float
    analytical: Dict[str, np.ndarray]


def run_hysteresis_sweep(
    sc: Optional[SimplicialComplex] = None,
    lambda_2: float = 2.8,
    lambda_1_grid: Optional[np.ndarray] = None,
    t_relax: float = 85.0,
    dt: float = 0.1,
    seed: int = 42,
) -> HysteresisSweepResult:
    """Perform adiabatic forward and backward sweeps of pairwise infectivity ``lambda_1``.

    Demonstrates continuous second-order transition when ``lambda_2 < 1`` (e.g. ``0.8``)
    versus explosive first-order tipping point and bistable hysteresis loop when
    ``lambda_2 > 1`` (e.g. ``2.8``).
    """
    if sc is None:
        sc = iacopini_rsc_complex(n_nodes=160, mean_k1=18.0, mean_k2=6.0, seed=seed)
    if lambda_1_grid is None:
        lambda_1_grid = np.linspace(0.20, 1.45, 34)
    else:
        lambda_1_grid = np.asarray(lambda_1_grid, dtype=np.float64)

    n_pts = len(lambda_1_grid)
    rho_fwd = np.zeros(n_pts, dtype=np.float64)
    rho_bwd = np.zeros(n_pts, dtype=np.float64)

    # Forward sweep: adiabatic continuation starting from small seed, keeping a tiny floor 2e-3
    p_curr = np.full(sc.num_nodes, 0.008, dtype=np.float64)
    for idx, l1 in enumerate(lambda_1_grid):
        p_seed = np.maximum(p_curr, 0.004)
        res = simulate_simplicial_contagion_ode(
            sc,
            lambda_1=float(l1),
            lambda_2=float(lambda_2),
            p_init=p_seed,
            t_max=t_relax,
            dt=dt,
            seed=seed + idx,
        )
        p_curr = res["p_final"]
        rho_fwd[idx] = float(np.mean(p_curr))

    # Backward sweep: adiabatic continuation starting from endemic high state
    p_curr_bwd = np.full(sc.num_nodes, 0.88, dtype=np.float64)
    for rev_idx in range(n_pts - 1, -1, -1):
        l1 = float(lambda_1_grid[rev_idx])
        p_seed_bwd = np.maximum(p_curr_bwd, 0.002)
        res_bwd = simulate_simplicial_contagion_ode(
            sc,
            lambda_1=l1,
            lambda_2=float(lambda_2),
            p_init=p_seed_bwd,
            t_max=t_relax,
            dt=dt,
            seed=seed + 1000 + rev_idx,
        )
        p_curr_bwd = res_bwd["p_final"]
        rho_bwd[rev_idx] = float(np.mean(p_curr_bwd))

    gap_curve = rho_bwd - rho_fwd
    max_idx = int(np.argmax(gap_curve))
    max_gap = float(gap_curve[max_idx])
    best_l1 = float(lambda_1_grid[max_idx])
    analytic = analytical_bifurcation_curve(lambda_1_grid, lambda_2=float(lambda_2))

    return HysteresisSweepResult(
        lambda_1_grid=lambda_1_grid,
        lambda_2=float(lambda_2),
        rho_forward=rho_fwd,
        rho_backward=rho_bwd,
        hysteresis_gap_curve=gap_curve,
        max_hysteresis_gap=max_gap,
        bistable_lambda_1_at_max_gap=best_l1,
        analytical=analytic,
    )
