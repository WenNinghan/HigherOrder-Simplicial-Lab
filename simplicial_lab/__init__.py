"""HigherOrder-Simplicial-Lab: Algebraic Topology, Simplicial Deep Learning & Contagion."""

from __future__ import annotations

from typing import Any

from .complexes import (
    SimplicialComplex,
    cycle_complex,
    iacopini_rsc_complex,
    planar_punctured_complex,
    random_simplicial_complex,
    trimolecular_reaction_complex,
)
from .contagion import (
    HysteresisSweepResult,
    analytical_bifurcation_curve,
    analytical_mean_field_equilibria,
    run_hysteresis_sweep,
    simulate_mean_field_ode,
    simulate_simplicial_contagion_ode,
    simulate_stochastic_contagion,
)
from .hodge import (
    HHDResult,
    cavity_filling_filtration,
    compute_betti_numbers,
    edge_cochain_to_node_vectors,
    edge_winding_operator,
    graph_laplacian_0,
    harmonic_basis,
    helmholtz_hodge_decomposition,
    hodge_laplacian_1,
    synthesize_hodge_flow,
)

__version__ = "0.1.0"

_MODEL_EXPORTS = {
    "HodgeSimplicialConvLayer",
    "HodgeSimplicialNN",
    "NodeGCNEdgePredictor",
    "EdgeLineGraphGCN",
    "EdgeMLP",
    "SimplicialTrajectoryClassifier",
    "build_torch_operators",
    "generate_edge_imputation_dataset",
    "compute_r2_and_mse",
    "train_and_eval_imputation",
    "generate_homotopy_trajectory_dataset",
    "train_and_eval_homotopy_classification",
}

__all__ = [
    "SimplicialComplex",
    "cycle_complex",
    "planar_punctured_complex",
    "random_simplicial_complex",
    "iacopini_rsc_complex",
    "trimolecular_reaction_complex",
    "graph_laplacian_0",
    "hodge_laplacian_1",
    "compute_betti_numbers",
    "harmonic_basis",
    "edge_winding_operator",
    "edge_cochain_to_node_vectors",
    "HHDResult",
    "helmholtz_hodge_decomposition",
    "synthesize_hodge_flow",
    "cavity_filling_filtration",
    "HodgeSimplicialConvLayer",
    "HodgeSimplicialNN",
    "NodeGCNEdgePredictor",
    "EdgeLineGraphGCN",
    "EdgeMLP",
    "SimplicialTrajectoryClassifier",
    "build_torch_operators",
    "generate_edge_imputation_dataset",
    "compute_r2_and_mse",
    "train_and_eval_imputation",
    "generate_homotopy_trajectory_dataset",
    "train_and_eval_homotopy_classification",
    "analytical_mean_field_equilibria",
    "analytical_bifurcation_curve",
    "simulate_mean_field_ode",
    "simulate_simplicial_contagion_ode",
    "simulate_stochastic_contagion",
    "HysteresisSweepResult",
    "run_hysteresis_sweep",
]


def __getattr__(name: str) -> Any:
    if name in _MODEL_EXPORTS:
        from . import models as _models

        val = getattr(_models, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'simplicial_lab' has no attribute {name!r}")
