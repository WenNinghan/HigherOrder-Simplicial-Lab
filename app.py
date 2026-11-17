"""Interactive Streamlit Workbench for HigherOrder-Simplicial-Lab.

Launch locally with:
    streamlit run app.py
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
import pandas as pd
import streamlit as st
import torch

from simplicial_lab import (
    HodgeSimplicialNN,
    NodeGCNEdgePredictor,
    analytical_bifurcation_curve,
    analytical_mean_field_equilibria,
    compute_betti_numbers,
    compute_r2_and_mse,
    edge_cochain_to_node_vectors,
    generate_edge_imputation_dataset,
    helmholtz_hodge_decomposition,
    hodge_laplacian_1,
    iacopini_rsc_complex,
    planar_punctured_complex,
    random_simplicial_complex,
    run_hysteresis_sweep,
    simulate_simplicial_contagion_ode,
    synthesize_hodge_flow,
    train_and_eval_imputation,
    trimolecular_reaction_complex,
)


st.set_page_config(
    page_title="HigherOrder-Simplicial-Lab Workbench",
    page_icon="🔺",
    layout="wide",
)


def _plot_complex_flow(
    ax: plt.Axes,
    sc,
    flow: np.ndarray,
    title: str,
    arrow_color: str = "#0f172a",
    cmap_name: str = "viridis",
) -> None:
    coords = sc.coords[:, :2]
    if sc.num_triangles > 0:
        verts = [coords[t] for t in sc.triangles]
        poly = PolyCollection(verts, facecolors="#e2e8f0", alpha=0.60, edgecolors="none", zorder=1)
        ax.add_collection(poly)

    segs = np.stack([coords[sc.edges[:, 0]], coords[sc.edges[:, 1]]], axis=1)
    mag = np.abs(flow)
    ref = max(float(np.percentile(mag, 97)), 1e-6) if len(mag) > 0 else 1.0
    s = np.clip(mag / ref, 0.0, 1.0)
    colors = plt.get_cmap(cmap_name)(0.15 + 0.80 * s)
    colors[:, 3] = 0.35 + 0.60 * s
    lc = LineCollection(segs, linewidths=0.6 + 2.1 * s, colors=colors, zorder=2)
    ax.add_collection(lc)

    ax.scatter(coords[:, 0], coords[:, 1], s=14, color="#334155", alpha=0.75, zorder=3)

    vecs = edge_cochain_to_node_vectors(sc, flow)
    vnorm = np.linalg.norm(vecs, axis=1)
    vref = max(float(np.percentile(vnorm, 95)), 1e-8)
    vscaled = (vecs / vref) * 0.13
    keep = vnorm > 0.08 * vref
    if np.any(keep):
        ax.quiver(
            coords[keep, 0],
            coords[keep, 1],
            vscaled[keep, 0],
            vscaled[keep, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=arrow_color,
            width=0.005,
            headwidth=3.8,
            zorder=4,
        )

    for idx, (cx, cy) in enumerate(sc.metadata.get("hole_centers", [])):
        r = sc.metadata.get("hole_radius", 0.24) * 0.70
        ax.add_patch(
            mpatches.Circle(
                (cx, cy),
                radius=r,
                facecolor="#ffffff",
                edgecolor="#991b1b",
                linestyle="--",
                linewidth=1.2,
                zorder=5,
            )
        )
        ax.text(cx, cy, f"Hole {'AB'[idx % 2]}", ha="center", va="center", fontsize=8, fontweight="bold", color="#7f1d1d", zorder=6)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10.5, fontweight="bold")


st.title("🔺 HigherOrder-Simplicial-Lab: Algebraic Topology & Simplicial Deep Learning")
st.markdown(
    "Interactive research workbench for **1-Hodge Laplacian Spectral Topology**, "
    "**Exact Helmholtz-Hodge Decomposition**, **Higher-Order Simplicial Contagion Hysteresis**, "
    "and **Orientation-Equivariant Hodge Simplicial Neural Networks**."
)

with st.sidebar:
    st.header("⚙️ Global Seed & Reference")
    global_seed = st.number_input("Random Seed", min_value=0, max_value=99999, value=42, step=1)
    st.markdown("---")
    st.markdown(
        "**Discrete Exterior Calculus**\n"
        "- Chain identity: `B1 @ B2 = 0`\n"
        "- 0-Laplacian: `L0 = B1 @ B1.T`\n"
        "- 1-Hodge: `L1 = B1.T @ B1 + B2 @ B2.T`\n"
        "- Hodge Split: `f = f_grad + f_curl + f_harm`"
    )

tab1, tab2, tab3 = st.tabs(
    [
        "1️⃣ Helmholtz-Hodge & Betti Topology Explorer",
        "2️⃣ Simplicial Contagion & Hysteresis Simulator",
        "3️⃣ Hodge-SNN vs Pairwise-GCN Flux Imputation",
    ]
)

# ===========================================================================
# TAB 1: Helmholtz-Hodge Flow Decomposition & Betti Topology Explorer
# ===========================================================================
with tab1:
    col_cfg, col_main = st.columns([1, 3])
    with col_cfg:
        complex_choice = st.selectbox(
            "Simplicial Complex Family",
            [
                "2-Hole Planar Punctured Mesh",
                "Trimolecular Reaction Complex (O3 + OH* + P)",
                "Random Simplicial Complex G_Delta",
            ],
        )
        w_grad = st.slider("Gradient Weight (w_grad)", 0.0, 2.0, 1.10, 0.05)
        w_curl = st.slider("Curl Weight (w_curl)", 0.0, 2.0, 1.00, 0.05)
        w_harm = st.slider("Harmonic Weight (w_harm)", 0.0, 2.0, 1.05, 0.05)

        if complex_choice == "Random Simplicial Complex G_Delta":
            rsc_n = st.slider("Nodes N0", 20, 65, 36, 2)
            rsc_p1 = st.slider("Edge Prob p1", 0.10, 0.40, 0.22, 0.02)
            rsc_p2 = st.slider("2-Simplex Fill Prob p2", 0.0, 1.0, 0.50, 0.05)
            sc_active = random_simplicial_complex(rsc_n, rsc_p1, rsc_p2, seed=int(global_seed))
        elif complex_choice.startswith("Trimolecular"):
            n_pol = st.slider("Pollutant Intermediates", 3, 10, 6, 1)
            sc_active = trimolecular_reaction_complex(n_pollutants=n_pol, seed=int(global_seed))
        else:
            sc_active = planar_punctured_complex(seed=int(global_seed))

    beta_0, beta_1 = compute_betti_numbers(sc_active, method="svd")
    synth = synthesize_hodge_flow(
        sc_active,
        grad_weight=w_grad,
        curl_weight=w_curl,
        harm_weight=w_harm,
        seed=int(global_seed),
    )
    hhd = helmholtz_hodge_decomposition(sc_active, synth["f_total"])

    with col_main:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Simplices (N0, N1, N2)", f"{sc_active.num_nodes}, {sc_active.num_edges}, {sc_active.num_triangles}")
        m2.metric("Betti (beta_0, beta_1)", f"({beta_0}, {beta_1})")
        m3.metric("||B1 @ B2||_max", f"{sc_active.verify_boundary_identity():.1e}")
        m4.metric("Max Orthogonality Err", f"{hhd.orthogonality_errors['max']:.2e}")
        m5.metric("Parseval Norm Err", f"{hhd.parseval_error:.2e}")

        fig1, axes1 = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
        _plot_complex_flow(
            axes1[0, 0],
            sc_active,
            hhd.f_total,
            f"Total Edge Cochain  (||f||^2 = {hhd.energies['total']:.2f})",
            cmap_name="magma_r",
        )
        _plot_complex_flow(
            axes1[0, 1],
            sc_active,
            hhd.f_grad,
            f"Gradient f_grad = B1^T phi  ({100*hhd.energy_fractions['grad']:.1f}%)",
            arrow_color="#1e3a8a",
            cmap_name="Blues",
        )
        _plot_complex_flow(
            axes1[1, 0],
            sc_active,
            hhd.f_curl,
            f"Curl f_curl = B2 psi  ({100*hhd.energy_fractions['curl']:.1f}%)",
            arrow_color="#581c87",
            cmap_name="Purples",
        )
        _plot_complex_flow(
            axes1[1, 1],
            sc_active,
            hhd.f_harm,
            f"Harmonic f_harm in ker(L1)  ({100*hhd.energy_fractions['harm']:.1f}%)",
            arrow_color="#7c2d12",
            cmap_name="YlOrRd",
        )
        st.pyplot(fig1)
        plt.close(fig1)

        l1_mat = hodge_laplacian_1(sc_active)
        evals_l1 = np.maximum(np.linalg.eigvalsh(l1_mat), 0.0)
        fig_spec, ax_spec = plt.subplots(figsize=(10.5, 2.4), constrained_layout=True)
        n_show = min(30, len(evals_l1))
        bar_colors = ["#dc2626" if ev < 1e-7 else "#2563eb" for ev in evals_l1[:n_show]]
        ax_spec.bar(np.arange(n_show), evals_l1[:n_show], color=bar_colors, edgecolor="#1e293b", linewidth=0.5)
        ax_spec.set_xlabel("Eigenvalue Index k")
        ax_spec.set_ylabel("lambda_k(L1)")
        ax_spec.set_title(f"Lowest {n_show} Eigenvalues of 1-Hodge Laplacian L1 (Red = Harmonic Zero-Modes, beta_1 = {beta_1})")
        st.pyplot(fig_spec)
        plt.close(fig_spec)

# ===========================================================================
# TAB 2: Higher-Order Simplicial Contagion & Hysteresis Simulator
# ===========================================================================
with tab2:
    c_left, c_right = st.columns([1, 3])
    with c_left:
        lam2 = st.slider("2-Simplex Synergistic Infectivity (lambda_2)", 0.2, 3.6, 2.8, 0.1)
        lam1_probe = st.slider("Probe Pairwise Infectivity (lambda_1)", 0.25, 1.40, 0.78, 0.02)
        rsc_nodes_sim = st.slider("Complex Size N0", 80, 220, 140, 20)

    @st.cache_data(show_spinner=False)
    def _cached_sweep(n0_val: int, l2_val: float, seed_val: int):
        sc_sim = iacopini_rsc_complex(n_nodes=n0_val, mean_k1=18.0, mean_k2=6.0, seed=seed_val)
        grid = np.linspace(0.20, 1.40, 26)
        sw = run_hysteresis_sweep(sc=sc_sim, lambda_2=l2_val, lambda_1_grid=grid, t_relax=80.0, seed=seed_val)
        return sc_sim, sw

    sc_sim, sweep_res = _cached_sweep(int(rsc_nodes_sim), float(lam2), int(global_seed))
    tip_back = float(sweep_res.analytical["lambda_1_critical_backward"])
    regime_str = "1st-Order Explosive (Bistable)" if lam2 > 1.0 else "2nd-Order Continuous"

    with c_right:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Transition Order", regime_str)
        k2.metric("Forward Threshold lambda_1^fwd", "1.000")
        k3.metric("Backward Tipping lambda_1^back", f"{tip_back:.3f}")
        k4.metric("Max Hysteresis Gap Delta rho", f"{sweep_res.max_hysteresis_gap:.3f}")

        fig2, (ax_bif, ax_ts) = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
        dense_x = np.linspace(0.20, 1.40, 250)
        ana = analytical_bifurcation_curve(dense_x, lambda_2=float(lam2))

        if lam2 > 1.0:
            ax_bif.axvspan(tip_back, 1.0, color="#fef2f2", alpha=0.85, label="Bistable Region")
        ax_bif.plot(dense_x, ana["upper_stable"], color="#991b1b", lw=2.1, label="Mean-Field Stable")
        ax_bif.plot(dense_x, ana["middle_unstable"], "--", color="#ef4444", lw=1.6, label="Unstable Separatrix")
        ax_bif.plot(sweep_res.lambda_1_grid, sweep_res.rho_forward, "o-", color="#dc2626", ms=4.8, label="Forward Sweep")
        ax_bif.plot(sweep_res.lambda_1_grid, sweep_res.rho_backward, "s-", color="#450a0a", ms=4.8, label="Backward Sweep")
        ax_bif.axvline(lam1_probe, color="#2563eb", linestyle=":", lw=1.6, label=f"Probe lambda_1={lam1_probe:.2f}")
        ax_bif.set_xlabel("Pairwise Infectivity lambda_1")
        ax_bif.set_ylabel("Steady-State Infected Fraction rho^*")
        ax_bif.set_title("Simplicial Bifurcation & Hysteresis Loop")
        ax_bif.legend(fontsize=8)
        ax_bif.grid(True, linestyle=":", alpha=0.45)

        eq_probe = analytical_mean_field_equilibria(float(lam1_probe), float(lam2))
        for r0, col, lbl in [
            (0.05, "#2563eb", "Low Seed (rho_0 = 0.05)"),
            (0.14, "#60a5fa", "Moderate Seed (rho_0 = 0.14)"),
            (0.32, "#f97316", "High Seed (rho_0 = 0.32)"),
            (0.65, "#b91c1c", "Endemic Seed (rho_0 = 0.65)"),
        ]:
            tr = simulate_simplicial_contagion_ode(
                sc_sim, lambda_1=float(lam1_probe), lambda_2=float(lam2), rho_init=r0, t_max=45.0, seed=int(global_seed)
            )
            ax_ts.plot(tr["times"], tr["rho"], color=col, lw=2.0, label=lbl)
        if eq_probe["unstable_separatrix"] is not None:
            ax_ts.axhline(eq_probe["unstable_separatrix"], color="#dc2626", linestyle="--", lw=1.4, label="Separatrix rho_-^*")
        ax_ts.set_xlabel("Dimensionless Time tau")
        ax_ts.set_ylabel("Infected Fraction rho(tau)")
        ax_ts.set_title(f"Trajectories at lambda_1 = {lam1_probe:.2f}, lambda_2 = {lam2:.1f}")
        ax_ts.set_ylim(-0.03, 0.90)
        ax_ts.legend(fontsize=8)
        ax_ts.grid(True, linestyle=":", alpha=0.45)

        st.pyplot(fig2)
        plt.close(fig2)

# ===========================================================================
# TAB 3: Hodge-SNN vs Pairwise-GCN Edge Flux Imputation Live Demo
# ===========================================================================
with tab3:
    d_left, d_right = st.columns([1, 3])
    with d_left:
        eta_pct = st.slider("Solenoidal (Curl + Harmonic) Energy %", 0, 100, 75, 5)
        mask_pct = st.slider("Missing Edge Mask %", 15, 60, 40, 5)
        epochs_demo = st.slider("Training Epochs", 40, 150, 90, 10)

    @st.cache_data(show_spinner=True)
    def _run_live_imputation(eta_val: float, m_ratio: float, ep: int, sd: int):
        sc_d = planar_punctured_complex(seed=sd)
        metrics = train_and_eval_imputation(
            sc_d,
            solenoidal_fraction=eta_val,
            mask_ratio=m_ratio,
            n_train=110,
            n_test=40,
            epochs=ep,
            hidden_channels=32,
            seed=sd,
        )
        return metrics

    metrics_live = _run_live_imputation(eta_pct / 100.0, mask_pct / 100.0, int(epochs_demo), int(global_seed))

    with d_right:
        df_live = pd.DataFrame(
            [
                {
                    "Architecture": m,
                    "Missing-Edge Test R^2": round(v["missing_r2"], 4),
                    "Missing-Edge Test MSE": round(v["missing_mse"], 4),
                    "Full-Complex R^2": round(v["overall_r2"], 4),
                }
                for m, v in metrics_live.items()
            ]
        )
        st.subheader(" Quantitative Missing Edge Flux Reconstruction")
        st.dataframe(df_live, use_container_width=True, hide_index=True)

        fig3, ax_bar = plt.subplots(figsize=(9.5, 3.6), constrained_layout=True)
        names = list(metrics_live.keys())
        r2_vals = [metrics_live[m]["missing_r2"] for m in names]
        cols = ["#2563eb", "#64748b", "#d97706", "#b91c1c"]
        bars = ax_bar.bar(names, r2_vals, color=cols, edgecolor="#1e293b", width=0.52)
        ax_bar.axhline(0.0, color="#94a3b8", linestyle="--", lw=1.0)
        for b, val in zip(bars, r2_vals):
            ax_bar.text(
                b.get_x() + b.get_width() / 2,
                max(val, 0.0) + 0.03,
                f"R^2 = {val:.3f}",
                ha="center",
                fontweight="bold",
                fontsize=9.5,
            )
        ax_bar.set_ylim(min(-0.15, min(r2_vals) - 0.1), 1.08)
        ax_bar.set_ylabel("Missing-Edge Test R^2")
        ax_bar.set_title(
            f"Imputation Performance at {eta_pct}% Solenoidal (Curl+Harmonic) Energy & {mask_pct}% Masked Edges"
        )
        ax_bar.grid(True, axis="y", linestyle=":", alpha=0.45)
        st.pyplot(fig3)
        plt.close(fig3)
