"""Rigorous Algebraic, Spectral, Deep Learning & Dynamical Unit Tests."""

from __future__ import annotations

import numpy as np
import torch

from simplicial_lab import (
    HodgeSimplicialNN,
    NodeGCNEdgePredictor,
    SimplicialComplex,
    compute_betti_numbers,
    cycle_complex,
    helmholtz_hodge_decomposition,
    hodge_laplacian_1,
    planar_punctured_complex,
    random_simplicial_complex,
    run_hysteresis_sweep,
    synthesize_hodge_flow,
    train_and_eval_homotopy_classification,
    train_and_eval_imputation,
    trimolecular_reaction_complex,
)


def test_exact_boundary_chain_identity() -> None:
    """Verify partial_1 o partial_2 = 0 (B1 @ B2 == 0) and L1_down @ L1_up == 0 identically."""
    complexes = [
        cycle_complex(n_nodes=7, filled=True),
        planar_punctured_complex(),
        random_simplicial_complex(n_nodes=45, p1=0.22, p2=0.60, seed=7),
        trimolecular_reaction_complex(n_pollutants=6, seed=11),
    ]
    for sc in complexes:
        assert sc.num_triangles > 0
        prod = sc.B1 @ sc.B2
        assert np.all(prod == 0.0), f"Non-zero boundary composition on {sc}: max={np.max(np.abs(prod))}"
        assert sc.verify_boundary_identity() == 0.0

        _, l1_down, l1_up = hodge_laplacian_1(sc, return_components=True)
        assert np.max(np.abs(l1_down @ l1_up)) < 1e-12


def test_exact_betti_numbers_across_topologies() -> None:
    """Verify exact (beta_0, beta_1) on cycles, filled disks, disconnected cycles, and 2-hole mesh."""
    # 1. Unfilled hexagon cycle -> beta_0 = 1, beta_1 = 1
    c6_open = cycle_complex(n_nodes=6, filled=False)
    assert compute_betti_numbers(c6_open, method="svd") == (1, 1)
    assert compute_betti_numbers(c6_open, method="spectral") == (1, 1)

    # 2. Single filled 2-simplex & fan-filled hexagon -> beta_0 = 1, beta_1 = 0
    tri_filled = SimplicialComplex(num_nodes=3, triangles=[(0, 1, 2)])
    assert compute_betti_numbers(tri_filled, method="svd") == (1, 0)
    assert compute_betti_numbers(tri_filled, method="spectral") == (1, 0)

    c6_filled = cycle_complex(n_nodes=6, filled=True)
    assert compute_betti_numbers(c6_filled, method="svd") == (1, 0)

    # 3. Disjoint union of two unfilled triangles -> beta_0 = 2, beta_1 = 2
    two_cycles = SimplicialComplex(
        num_nodes=6,
        edges=[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)],
        triangles=[],
    )
    assert compute_betti_numbers(two_cycles, method="svd") == (2, 2)
    assert compute_betti_numbers(two_cycles, method="spectral") == (2, 2)

    # 4. 2-hole planar punctured complex -> beta_0 = 1, beta_1 = 2
    punctured = planar_punctured_complex()
    assert compute_betti_numbers(punctured, method="svd") == (1, 2)
    assert compute_betti_numbers(punctured, method="spectral") == (1, 2)


def test_helmholtz_hodge_orthogonality_and_parseval() -> None:
    """Verify pairwise orthogonality (<1e-10), Parseval energy conservation, and kernel memberships."""
    sc = planar_punctured_complex()
    rng = np.random.default_rng(123)

    # Test both random unstructured 1-cochains and synthesized physical multi-component flows
    flows = [
        rng.standard_normal(sc.num_edges),
        synthesize_hodge_flow(sc, grad_weight=1.4, curl_weight=1.1, harm_weight=0.9, seed=99)["f_total"],
    ]
    for f in flows:
        hhd = helmholtz_hodge_decomposition(sc, f)
        assert np.max(np.abs(hhd.f_total - (hhd.f_grad + hhd.f_curl + hhd.f_harm))) < 1e-12
        assert hhd.orthogonality_errors["grad_curl"] < 1e-10
        assert hhd.orthogonality_errors["grad_harm"] < 1e-10
        assert hhd.orthogonality_errors["curl_harm"] < 1e-10
        assert hhd.parseval_error < 1e-10

        # Exact Discrete Exterior Calculus kernel identities:
        # curl(grad) = B2.T @ f_grad == 0, div(curl) = B1 @ f_curl == 0, and harmonic in ker(B1) cap ker(B2.T)
        assert np.max(np.abs(sc.B2.T @ hhd.f_grad)) < 1e-10
        assert np.max(np.abs(sc.B1 @ hhd.f_curl)) < 1e-10
        assert np.max(np.abs(sc.B1 @ hhd.f_harm)) < 1e-10
        assert np.max(np.abs(sc.B2.T @ hhd.f_harm)) < 1e-10


def test_orientation_equivariance_and_solenoidal_expressivity() -> None:
    """Verify model(-x) == -model(x) and HodgeSimplicialNN >> NodeGCN on solenoidal flows."""
    sc = planar_punctured_complex()
    torch.manual_seed(42)
    hodge_net = HodgeSimplicialNN(sc, hidden_channels=24, num_layers=3, equivariant=True).double()
    node_net = NodeGCNEdgePredictor(sc, hidden_channels=24, num_layers=3, equivariant=True).double()
    hodge_net.eval()
    node_net.eval()

    x = torch.randn(sc.num_edges, 1, dtype=torch.float64)
    mask = (torch.rand(sc.num_edges, 1, dtype=torch.float64) > 0.4).double()

    with torch.no_grad():
        out_pos = hodge_net(x)
        out_neg = hodge_net(-x)
        assert torch.max(torch.abs(out_neg + out_pos)).item() < 1e-12

        out_m_pos = hodge_net(x, mask=mask)
        out_m_neg = hodge_net(-x, mask=mask)
        assert torch.max(torch.abs(out_m_neg + out_m_pos)).item() < 1e-12

        # NodeGCN output must lie strictly inside im(B1.T) (zero curl and zero harmonic energy!)
        y_node = node_net(x).squeeze(-1).numpy()
        hhd_node = helmholtz_hodge_decomposition(sc, y_node)
        assert hhd_node.energies["curl"] < 1e-20
        assert hhd_node.energies["harm"] < 1e-20

    # Empirical benchmark: solenoidal (curl + harmonic) missing edge flux imputation
    metrics = train_and_eval_imputation(
        sc,
        solenoidal_fraction=0.85,
        mask_ratio=0.40,
        n_train=90,
        n_test=36,
        epochs=80,
        hidden_channels=24,
        seed=42,
    )
    r2_hodge = metrics["HodgeSimplicialNN"]["missing_r2"]
    r2_node = metrics["NodeGCN"]["missing_r2"]
    assert r2_hodge > r2_node + 0.35, (
        f"Expected HodgeSimplicialNN ({r2_hodge:.3f}) to strongly outperform NodeGCN ({r2_node:.3f})."
    )


def test_homotopy_trajectory_classification_advantage() -> None:
    """Verify HodgeSimplicialNN outperforms NodeGCN on cavity homotopy loop classification."""
    sc = planar_punctured_complex()
    acc = train_and_eval_homotopy_classification(
        sc,
        n_train_per_class=28,
        n_test_per_class=20,
        epochs=75,
        hidden_channels=24,
        seed=42,
    )
    assert acc["HodgeSimplicialNN"] > 0.80
    assert acc["HodgeSimplicialNN"] > acc["NodeGCN"] + 0.30


def test_higher_order_contagion_bistability_and_hysteresis() -> None:
    """Verify discontinuous hysteresis gap > 0.25 at lambda_2=2.8 vs continuous transition at lambda_2=0.8."""
    sweep_bistable = run_hysteresis_sweep(
        lambda_2=2.8,
        lambda_1_grid=np.linspace(0.30, 1.35, 22),
        t_relax=80.0,
        seed=42,
    )
    assert sweep_bistable.max_hysteresis_gap > 0.25, (
        f"Expected bistable hysteresis gap > 0.25, got {sweep_bistable.max_hysteresis_gap:.4f}"
    )

    sweep_continuous = run_hysteresis_sweep(
        lambda_2=0.8,
        lambda_1_grid=np.linspace(0.30, 1.35, 22),
        t_relax=95.0,
        seed=42,
    )
    assert sweep_continuous.max_hysteresis_gap < 0.08
