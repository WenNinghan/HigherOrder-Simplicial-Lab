"""Automated Benchmark Suite & Top-Journal Publication Figure Generator (300 DPI).

Executes all topological, dynamical, and geometric deep learning benchmarks and writes:
- ``results/benchmark_summary.json``
- ``results/benchmark_table.csv``
- ``assets/helmholtz_hodge_decomposition.png``
- ``assets/simplicial_phase_transition.png``
- ``assets/hodge_snn_benchmark.png``
- ``assets/hodge_spectrum_cavity_filling.png``
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if hasattr(platform, "_wmi"):
    platform._wmi = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_linear_imputation import evaluate_linear_control
from simplicial_lab import (
    HodgeSimplicialNN,
    NodeGCNEdgePredictor,
    analytical_mean_field_equilibria,
    cavity_filling_filtration,
    compute_betti_numbers,
    edge_cochain_to_node_vectors,
    helmholtz_hodge_decomposition,
    iacopini_rsc_complex,
    planar_punctured_complex,
    run_hysteresis_sweep,
    simulate_simplicial_contagion_ode,
    simulate_stochastic_contagion,
    synthesize_hodge_flow,
    train_and_eval_homotopy_classification,
    train_and_eval_imputation,
    trimolecular_reaction_complex,
)


def _setup_plot_style() -> None:
    """Configure SIAM Review / Nature Communications 300-DPI bold serif & STIX math style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "font.weight": "bold",
            "mathtext.fontset": "stix",
            "font.size": 10.2,
            "axes.titlesize": 11.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "legend.fontsize": 8.4,
            "figure.titlesize": 12.6,
            "figure.titleweight": "bold",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "axes.edgecolor": "#111111",
            "axes.linewidth": 1.30,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _sci_tex(val: float) -> str:
    """Format floating-point value as clean STIX scientific notation a.bc x 10^{exp}."""
    if val == 0.0:
        return "0"
    mant_str, exp_str = f"{val:.2e}".split("e")
    exp_int = int(exp_str)
    return f"{mant_str} \\times 10^{{{exp_int}}}"


def _style_continuous_axes(ax: plt.Axes) -> None:
    """Apply 4-sided closed #111111 frame, 4-sided inward ticks, and subtle dotted grid."""
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#111111")
        spine.set_linewidth(1.30)
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        bottom=True,
        left=True,
        length=4.5,
        width=1.2,
        color="#111111",
        labelcolor="#111111",
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        bottom=True,
        left=True,
        length=2.5,
        width=0.9,
        color="#111111",
    )
    ax.grid(True, which="major", linestyle=":", linewidth=0.70, alpha=0.35, color="#94A3B8")


def _style_categorical_axes(ax: plt.Axes) -> None:
    """Apply 4-sided closed #111111 frame, inward y-ticks, and outward bottom x-ticks."""
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#111111")
        spine.set_linewidth(1.30)
    ax.tick_params(
        axis="y",
        which="major",
        direction="in",
        left=True,
        right=True,
        length=4.5,
        width=1.2,
        color="#111111",
        labelcolor="#111111",
    )
    ax.tick_params(
        axis="x",
        which="major",
        direction="out",
        bottom=True,
        top=False,
        length=4.2,
        width=1.2,
        color="#111111",
        labelcolor="#111111",
    )
    ax.grid(True, axis="y", linestyle=":", linewidth=0.70, alpha=0.35, color="#94A3B8")
    ax.grid(False, axis="x")


def _finalize_journal_figure(fig: plt.Figure) -> None:
    """Enforce bold Times New Roman across every axis, colorbar, tick, annotation, and legend."""
    fig.canvas.draw()
    for text_obj in fig.texts:
        text_obj.set_fontfamily("Times New Roman")
        text_obj.set_fontweight("bold")

    for ax in fig.axes:
        ax.title.set_fontfamily("Times New Roman")
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontfamily("Times New Roman")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontfamily("Times New Roman")
        ax.yaxis.label.set_fontweight("bold")
        ax.xaxis.get_offset_text().set_fontfamily("Times New Roman")
        ax.xaxis.get_offset_text().set_fontweight("bold")
        ax.yaxis.get_offset_text().set_fontfamily("Times New Roman")
        ax.yaxis.get_offset_text().set_fontweight("bold")

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily("Times New Roman")
            lbl.set_fontweight("bold")
        for lbl in ax.get_xticklabels(minor=True) + ax.get_yticklabels(minor=True):
            lbl.set_fontfamily("Times New Roman")
            lbl.set_fontweight("bold")

        for txt in ax.texts:
            txt.set_fontfamily("Times New Roman")
            txt.set_fontweight("bold")

        leg = ax.get_legend()
        if leg is not None:
            for ltxt in leg.get_texts():
                ltxt.set_fontfamily("Times New Roman")
                ltxt.set_fontweight("bold")
            if leg.get_title() is not None:
                leg.get_title().set_fontfamily("Times New Roman")
                leg.get_title().set_fontweight("bold")
            frame = leg.get_frame()
            frame.set_linewidth(1.1)
            frame.set_edgecolor("#111111")
            frame.set_facecolor("#FFFFFF")


def _draw_simplicial_flow_panel(
    ax: plt.Axes,
    sc: Any,
    flow: np.ndarray,
    title: str,
    subtitle: str,
    tri_values: np.ndarray | None = None,
    node_values: np.ndarray | None = None,
    tri_cmap: str = "coolwarm",
    arrow_color: str = "#111111",
    edge_cmap: str = "viridis",
) -> None:
    coords = sc.coords[:, :2]
    ax.set_facecolor("#FFFFFF")

    # 1. Draw 2-simplices
    if sc.num_triangles > 0:
        verts = [coords[tri] for tri in sc.triangles]
        if tri_values is not None:
            vmax = max(float(np.max(np.abs(tri_values))), 1e-6)
            norm = plt.Normalize(vmin=-vmax, vmax=vmax)
            facecolors = plt.get_cmap(tri_cmap)(norm(tri_values))
            facecolors[:, 3] = 0.50
            poly = PolyCollection(verts, facecolors=facecolors, edgecolors="none", zorder=1)
        else:
            poly = PolyCollection(
                verts,
                facecolors="#EAECEE",
                alpha=0.58,
                edgecolors="none",
                zorder=1,
            )
        ax.add_collection(poly)

    # 2. Draw 1-simplices weighted by |flow|
    segments = np.stack([coords[sc.edges[:, 0]], coords[sc.edges[:, 1]]], axis=1)
    abs_f = np.abs(flow)
    f_max = max(float(np.percentile(abs_f, 98)), 1e-6)
    strength = np.clip(abs_f / f_max, 0.0, 1.0)
    widths = 0.55 + 2.25 * strength
    colors = plt.get_cmap(edge_cmap)(0.18 + 0.78 * strength)
    colors[:, 3] = 0.36 + 0.60 * strength
    lc = LineCollection(segments, linewidths=widths, colors=colors, zorder=2)
    ax.add_collection(lc)

    # 3. Optional scalar node potential shading
    if node_values is not None:
        vmax_n = max(float(np.max(np.abs(node_values))), 1e-6)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=node_values,
            cmap="RdBu_r",
            vmin=-vmax_n,
            vmax=vmax_n,
            s=28,
            edgecolors="#111111",
            linewidths=0.50,
            zorder=4,
        )
    else:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            color="#2C3E50",
            s=12,
            alpha=0.72,
            zorder=3,
        )

    # 4. Draw interpolated quiver arrows at vertices (clipped cleanly inside panel frame)
    node_vecs = edge_cochain_to_node_vectors(sc, flow)
    vec_norms = np.linalg.norm(node_vecs, axis=1)
    v_ref = max(float(np.percentile(vec_norms, 95)), 1e-8)
    scaled_vecs = (node_vecs / v_ref) * 0.125
    keep = vec_norms > 0.08 * v_ref
    ax.quiver(
        coords[keep, 0],
        coords[keep, 1],
        scaled_vecs[keep, 0],
        scaled_vecs[keep, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=arrow_color,
        width=0.0046,
        headwidth=3.7,
        headlength=4.4,
        alpha=0.92,
        clip_on=True,
        zorder=5,
    )

    # 5. Annotate topological holes
    hole_centers = sc.metadata.get("hole_centers", [])
    hole_r = sc.metadata.get("hole_radius", 0.24) * 0.72
    hole_names = ["Hole A", "Hole B"]
    for idx, (cx, cy) in enumerate(hole_centers):
        circ = mpatches.Circle(
            (cx, cy),
            radius=hole_r,
            facecolor="#FFFFFF",
            edgecolor="#B03A2E",
            linestyle="--",
            linewidth=1.35,
            alpha=0.95,
            zorder=6,
        )
        ax.add_patch(circ)
        lbl = hole_names[idx] if idx < len(hole_names) else f"Hole {idx+1}"
        ax.text(
            cx,
            cy,
            lbl,
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="#B03A2E",
            zorder=7,
        )

    # Generous margin so boundary quiver arrowheads stay strictly inside closed 1.2pt border
    ax.set_xlim(-1.19, 1.19)
    ax.set_ylim(-0.86, 0.86)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#111111")
        spine.set_linewidth(1.20)
    ax.set_title(title, pad=20, fontsize=11.1, fontweight="bold")
    ax.text(
        0.5,
        1.022,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.7,
        color="#1B2631",
    )


def generate_figure_1_hhd(assets_dir: Path) -> Dict[str, Any]:
    """Generate 4-panel Helmholtz-Hodge Decomposition visualization on 2-hole complex."""
    sc = planar_punctured_complex(seed=42)
    beta_0, beta_1 = compute_betti_numbers(sc, method="svd")
    synth = synthesize_hodge_flow(
        sc, grad_weight=1.15, curl_weight=1.00, harm_weight=1.05, seed=42
    )
    hhd = helmholtz_hodge_decomposition(sc, synth["f_total"])

    div_curl = float(np.max(np.abs(sc.B1 @ hhd.f_curl)))
    curl_grad = float(np.max(np.abs(sc.B2.T @ hhd.f_grad)))
    l1_harm = float(
        np.max(np.abs((sc.B1.T @ (sc.B1 @ hhd.f_harm)) + (sc.B2 @ (sc.B2.T @ hhd.f_harm))))
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6), constrained_layout=True)

    _draw_simplicial_flow_panel(
        axes[0, 0],
        sc,
        hhd.f_total,
        title="(a) Total 1-Cochain Edge Flux $\\mathbf{f} = \\mathbf{f}_{\\mathrm{grad}} + \\mathbf{f}_{\\mathrm{curl}} + \\mathbf{f}_{\\mathrm{harm}}$",
        subtitle=(
            f"$N_0 = {sc.num_nodes},\\; N_1 = {sc.num_edges},\\; N_2 = {sc.num_triangles}$  |  "
            f"Betti $(\\beta_0 = {beta_0},\\; \\beta_1 = {beta_1})$  |  "
            f"$||\\mathbf{{f}}||_2^2 = {hhd.energies['total']:.3f}$"
        ),
        edge_cmap="magma_r",
        arrow_color="#111111",
    )

    _draw_simplicial_flow_panel(
        axes[0, 1],
        sc,
        hhd.f_grad,
        title="(b) Irrotational Gradient Component $\\mathbf{f}_{\\mathrm{grad}} = \\mathbf{B}_1^\\top \\boldsymbol{\\phi}$",
        subtitle=(
            f"Energy Share: {100.0 * hhd.energy_fractions['grad']:.1f}%  |  "
            f"Curl-Free Residual $||\\mathbf{{B}}_2^\\top \\mathbf{{f}}_{{\\mathrm{{grad}}}}||_\\infty = {_sci_tex(curl_grad)}$"
        ),
        node_values=hhd.phi - np.mean(hhd.phi),
        edge_cmap="Blues",
        arrow_color="#1B4F72",
    )

    _draw_simplicial_flow_panel(
        axes[1, 0],
        sc,
        hhd.f_curl,
        title="(c) Coexact Rotational Curl Component $\\mathbf{f}_{\\mathrm{curl}} = \\mathbf{B}_2 \\boldsymbol{\\psi}$",
        subtitle=(
            f"Energy Share: {100.0 * hhd.energy_fractions['curl']:.1f}%  |  "
            f"Div-Free Residual $||\\mathbf{{B}}_1 \\mathbf{{f}}_{{\\mathrm{{curl}}}}||_\\infty = {_sci_tex(div_curl)}$"
        ),
        tri_values=hhd.psi,
        tri_cmap="PiYG",
        edge_cmap="Purples",
        arrow_color="#5B2C6F",
    )

    _draw_simplicial_flow_panel(
        axes[1, 1],
        sc,
        hhd.f_harm,
        title="(d) Topological Harmonic Component $\\mathbf{f}_{\\mathrm{harm}} \\in \\ker(\\mathbf{L}_1)$",
        subtitle=(
            f"Energy Share: {100.0 * hhd.energy_fractions['harm']:.1f}%  |  "
            f"Hodge Kernel Residual $||\\mathbf{{L}}_1 \\mathbf{{f}}_{{\\mathrm{{harm}}}}||_\\infty = {_sci_tex(l1_harm)}$"
        ),
        edge_cmap="YlOrRd",
        arrow_color="#B03A2E",
    )

    fig.suptitle(
        "Orthogonal Helmholtz–Hodge Decomposition ($\\mathbf{L}_1 = \\mathbf{B}_1^\\top \\mathbf{B}_1 + \\mathbf{B}_2 \\mathbf{B}_2^\\top$;  "
        f"Max Ortho. Error $= {_sci_tex(hhd.orthogonality_errors['max'])}$,  "
        f"Parseval Error $= {_sci_tex(hhd.parseval_error)}$)",
        fontsize=12.3,
        fontweight="bold",
    )

    _finalize_journal_figure(fig)
    out_path = assets_dir / "helmholtz_hodge_decomposition.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "complex_shape": {"N0": sc.num_nodes, "N1": sc.num_edges, "N2": sc.num_triangles},
        "betti_numbers": {"beta_0": beta_0, "beta_1": beta_1},
        "boundary_identity_error": sc.verify_boundary_identity(),
        "energies": hhd.energies,
        "energy_fractions": hhd.energy_fractions,
        "orthogonality_errors": hhd.orthogonality_errors,
        "parseval_error": hhd.parseval_error,
        "kernel_residuals": {
            "curl_of_grad_linf": curl_grad,
            "div_of_curl_linf": div_curl,
            "l1_of_harm_linf": l1_harm,
        },
    }


def generate_figure_2_contagion(assets_dir: Path) -> Dict[str, Any]:
    """Generate 2-panel Simplicial Contagion Hysteresis & Bistability figure."""
    sc_rsc = iacopini_rsc_complex(n_nodes=180, mean_k1=18.0, mean_k2=6.0, seed=42)
    k1_emp, k2_emp = sc_rsc.mean_degrees()
    l1_grid = np.linspace(0.20, 1.45, 36)

    sweep_exp = run_hysteresis_sweep(
        sc=sc_rsc, lambda_2=2.8, lambda_1_grid=l1_grid, t_relax=95.0, seed=42
    )
    sweep_cont = run_hysteresis_sweep(
        sc=sc_rsc, lambda_2=0.8, lambda_1_grid=l1_grid, t_relax=95.0, seed=42
    )

    dense_l1 = np.linspace(0.20, 1.45, 300)
    from simplicial_lab.contagion import analytical_bifurcation_curve

    ana_exp = analytical_bifurcation_curve(dense_l1, lambda_2=2.8)
    ana_cont = analytical_bifurcation_curve(dense_l1, lambda_2=0.8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.6), constrained_layout=True)
    _style_continuous_axes(ax1)
    _style_continuous_axes(ax2)

    # Shade bistable hysteresis zone
    tip_back = float(ana_exp["lambda_1_critical_backward"])
    ax1.axvspan(
        tip_back,
        1.0,
        color="#FDEDEC",
        alpha=0.72,
        zorder=0,
        label=f"Bistable Window $[{tip_back:.2f},\\; 1.00]$",
    )

    # Analytical branches
    ax1.plot(
        dense_l1,
        ana_exp["upper_stable"],
        color="#B03A2E",
        linewidth=2.4,
        zorder=3,
        label="Mean-Field Stable ($\\lambda_2 = 2.8$)",
    )
    ax1.plot(
        dense_l1,
        ana_exp["middle_unstable"],
        color="#C55A11",
        linestyle="--",
        linewidth=1.9,
        zorder=3,
        label="Unstable Separatrix $\\rho_-^*$ ($\\lambda_2 = 2.8$)",
    )
    ax1.plot(
        dense_l1,
        ana_cont["upper_stable"],
        color="#0B6E4F",
        linewidth=2.2,
        zorder=3,
        label="Mean-Field Continuous ($\\lambda_2 = 0.8$)",
    )

    # Empirical simplicial complex sweeps
    ax1.plot(
        l1_grid,
        sweep_exp.rho_forward,
        "o-",
        color="#C55A11",
        markersize=5.0,
        markeredgecolor="#111111",
        markeredgewidth=0.45,
        linewidth=1.35,
        alpha=0.92,
        zorder=4,
        label="Complex Forward Sweep ($\\lambda_2 = 2.8$)",
    )
    ax1.plot(
        l1_grid,
        sweep_exp.rho_backward,
        "s-",
        color="#5B2C6F",
        markersize=4.8,
        markeredgecolor="#111111",
        markeredgewidth=0.45,
        linewidth=1.35,
        alpha=0.92,
        zorder=5,
        label="Complex Backward Sweep ($\\lambda_2 = 2.8$)",
    )
    ax1.plot(
        l1_grid,
        sweep_cont.rho_forward,
        "^-",
        color="#1B4F72",
        markersize=5.0,
        markeredgecolor="#111111",
        markeredgewidth=0.45,
        linewidth=1.35,
        alpha=0.92,
        zorder=4,
        label="Complex Sweep ($\\lambda_2 = 0.8$, 2nd-Order)",
    )

    # Annotate hysteresis gap
    l1_star = sweep_exp.bistable_lambda_1_at_max_gap
    idx_star = int(np.argmin(np.abs(l1_grid - l1_star)))
    y_lo = sweep_exp.rho_forward[idx_star]
    y_hi = sweep_exp.rho_backward[idx_star]
    ax1.annotate(
        "",
        xy=(l1_star, y_hi),
        xytext=(l1_star, y_lo),
        arrowprops=dict(arrowstyle="<->", color="#B03A2E", lw=1.85),
        zorder=6,
    )
    ax1.text(
        l1_star - 0.03,
        0.5 * (y_lo + y_hi),
        f"$\\Delta\\rho = {sweep_exp.max_hysteresis_gap:.2f}$",
        ha="right",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#B03A2E",
        zorder=7,
        bbox=dict(boxstyle="square,pad=0.25", facecolor="#FFFFFF", edgecolor="#B03A2E", lw=1.0, alpha=0.96),
    )

    ax1.set_xlabel("Rescaled Pairwise Infectivity $\\lambda_1 = \\beta_1 \\langle k_1 \\rangle / \\mu$")
    ax1.set_ylabel("Steady-State Infected Density $\\rho^*$")
    ax1.set_title("(a) Explosive First-Order Tipping & Hysteresis vs. Pairwise Transition", pad=8)
    ax1.set_xlim(0.20, 1.45)
    # Generous upper headroom so upper-left legend clears backward sweep curve completely
    ax1.set_ylim(-0.03, 1.05)
    ax1.legend(
        loc="upper left",
        fancybox=False,
        edgecolor="#111111",
        framealpha=0.96,
        fontsize=7.9,
    )

    # Panel (b): Time-series bistability splitting at lambda_1 = 0.78, lambda_2 = 2.8
    l1_bistable = 0.78
    eq_bi = analytical_mean_field_equilibria(l1_bistable, 2.8)
    rho_sep = float(eq_bi["unstable_separatrix"] or 0.16)
    rho_end = float(eq_bi["stable_endemic"] or 0.56)

    seeds_rho0 = [0.04, 0.09, 0.13, 0.19, 0.25, 0.36, 0.50, 0.74]
    cmap_low = plt.get_cmap("Blues")
    cmap_high = plt.get_cmap("Reds")

    for s_idx, r0 in enumerate(seeds_rho0):
        traj = simulate_simplicial_contagion_ode(
            sc_rsc,
            lambda_1=l1_bistable,
            lambda_2=2.8,
            rho_init=r0,
            t_max=45.0,
            dt=0.08,
            seed=100 + s_idx,
        )
        is_upper = r0 > rho_sep
        color = (
            cmap_high(0.58 + 0.34 * (s_idx / len(seeds_rho0)))
            if is_upper
            else cmap_low(0.58 + 0.34 * (s_idx / 3.0))
        )
        lbl = None
        if r0 == seeds_rho0[0]:
            lbl = "Sub-threshold Seeds ($\\rho_0 < \\rho_-^* \\to 0$)"
        elif r0 == seeds_rho0[-1]:
            lbl = "Super-threshold Seeds ($\\rho_0 > \\rho_-^* \\to \\rho_+^*$)"
        ax2.plot(traj["times"], traj["rho"], color=color, linewidth=2.05, zorder=4, label=lbl)

    # Overlay two stochastic Markovian realizations (high vs low initial seed)
    stoch_hi = simulate_stochastic_contagion(
        sc_rsc, lambda_1=l1_bistable, lambda_2=2.8, rho_init=0.32, t_max=45.0, spontaneous_rate=0.0, seed=7
    )
    stoch_lo = simulate_stochastic_contagion(
        sc_rsc, lambda_1=l1_bistable, lambda_2=2.8, rho_init=0.10, t_max=45.0, spontaneous_rate=0.0, seed=8
    )
    ax2.plot(
        stoch_hi["times"],
        stoch_hi["rho"],
        color="#B03A2E",
        alpha=0.40,
        linewidth=1.15,
        zorder=2,
        label="Microscopic Markov Realizations",
    )
    ax2.plot(stoch_lo["times"], stoch_lo["rho"], color="#1B4F72", alpha=0.40, linewidth=1.15, zorder=2)

    ax2.axhline(
        rho_sep,
        color="#C55A11",
        linestyle="--",
        linewidth=1.75,
        zorder=3,
        label=f"Unstable Separatrix $\\rho_-^* = {rho_sep:.3f}$",
    )
    ax2.axhline(
        rho_end,
        color="#5B2C6F",
        linestyle="-.",
        linewidth=1.60,
        zorder=3,
        label=f"Endemic Attractor $\\rho_+^* = {rho_end:.3f}$",
    )

    ax2.set_xlabel("Dimensionless Time $\\tau = \\mu t$")
    ax2.set_ylabel("Active / Infected Fraction $\\rho(\\tau)$")
    ax2.set_title(
        f"(b) Bistable Basin Splitting ($\\lambda_1 = {l1_bistable:.2f},\\; \\lambda_2 = 2.8$)",
        pad=8,
    )
    ax2.set_xlim(0.0, 45.0)
    # Upper headroom places upper-right legend completely above stochastic realizations (max ~0.68)
    ax2.set_ylim(-0.03, 1.00)
    ax2.legend(
        loc="upper right",
        fancybox=False,
        edgecolor="#111111",
        framealpha=0.96,
        fontsize=8.0,
    )

    fig.suptitle(
        f"Higher-Order Simplicial Contagion Dynamics on Random 2-Complex "
        f"($N = {sc_rsc.num_nodes},\\; \\langle k_1 \\rangle = {k1_emp:.1f},\\; \\langle k_2 \\rangle = {k2_emp:.1f}$)",
        fontsize=12.6,
        fontweight="bold",
    )

    _finalize_journal_figure(fig)
    out_path = assets_dir / "simplicial_phase_transition.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "rsc_nodes": sc_rsc.num_nodes,
        "mean_k1": k1_emp,
        "mean_k2": k2_emp,
        "lambda_2_bistable": 2.8,
        "lambda_2_continuous": 0.8,
        "critical_lambda_1_forward": 1.0,
        "critical_lambda_1_backward_analytical": tip_back,
        "max_hysteresis_gap_bistable": sweep_exp.max_hysteresis_gap,
        "bistable_lambda_1_at_max_gap": sweep_exp.bistable_lambda_1_at_max_gap,
        "max_hysteresis_gap_continuous": sweep_cont.max_hysteresis_gap,
        "bistable_separatrix_at_078": rho_sep,
        "bistable_endemic_at_078": rho_end,
    }


def generate_figure_3_snn_benchmarks(assets_dir: Path) -> Dict[str, Any]:
    """Run Missing Edge Flux Imputation & Homotopy Trajectory Classification benchmarks."""
    sc = planar_punctured_complex(seed=42)
    eta_grid = [0.0, 0.25, 0.50, 0.75, 1.00]
    model_names = ["NodeGCN", "EdgeMLP", "Isotropic-L1-GCN", "HodgeSimplicialNN"]

    # Check exact global and randomized local D=diag(+-1) orientation equivariance
    torch.manual_seed(42)
    rng_eq = np.random.default_rng(42)
    rand_signs = rng_eq.choice([-1.0, 1.0], size=sc.num_edges)

    net_eq = HodgeSimplicialNN(sc, equivariant=True, dtype=torch.float64).eval()
    net_eq_d = net_eq.with_edge_orientations(sc, rand_signs)
    x_probe = torch.randn(sc.num_edges, 1, dtype=torch.float64)
    m_probe = (torch.rand(sc.num_edges, 1, dtype=torch.float64) > 0.4).double()
    d_col64 = torch.from_numpy(rand_signs).double().unsqueeze(-1)

    net_eq32 = HodgeSimplicialNN(sc, equivariant=True, dtype=torch.float32).eval()
    net_eq32_d = net_eq32.with_edge_orientations(sc, rand_signs)
    x_probe32 = x_probe.float()
    m_probe32 = m_probe.float()
    d_col32 = d_col64.float()

    with torch.no_grad():
        equiv_err = float(torch.max(torch.abs(net_eq(-x_probe) + net_eq(x_probe))).item())
        gauge_err_f64 = float(
            torch.max(
                torch.abs(net_eq_d(d_col64 * x_probe, mask=m_probe) - d_col64 * net_eq(x_probe, mask=m_probe))
            ).item()
        )
        gauge_err_f32 = float(
            torch.max(
                torch.abs(net_eq32_d(d_col32 * x_probe32, mask=m_probe32) - d_col32 * net_eq32(x_probe32, mask=m_probe32))
            ).item()
        )

    imputation_curves: Dict[str, Dict[str, List[float]]] = {
        m: {"r2": [], "mse": []} for m in model_names
    }

    for eta in eta_grid:
        res_eta = train_and_eval_imputation(
            sc,
            solenoidal_fraction=eta,
            mask_ratio=0.40,
            n_train=150,
            n_test=60,
            epochs=130,
            hidden_channels=32,
            lr=3e-3,
            seed=42,
        )
        for m in model_names:
            imputation_curves[m]["r2"].append(res_eta[m]["missing_r2"])
            imputation_curves[m]["mse"].append(res_eta[m]["missing_mse"])

    # Multi-seed homotopy trajectory classification
    seeds = [42, 101, 202]
    classification_models = [*model_names, "HarmonicCentroid"]
    homotopy_runs: Dict[str, List[float]] = {m: [] for m in classification_models}
    for sd in seeds:
        acc_sd = train_and_eval_homotopy_classification(
            sc,
            n_train_per_class=45,
            n_test_per_class=30,
            epochs=120,
            hidden_channels=32,
            lr=3e-3,
            seed=sd,
        )
        for m in classification_models:
            homotopy_runs[m].append(acc_sd[m])

    homotopy_mean = {m: float(np.mean(runs)) for m, runs in homotopy_runs.items()}
    homotopy_std = {m: float(np.std(runs)) for m, runs in homotopy_runs.items()}

    summary = {
        "orientation_equivariance_max_error_float64": equiv_err,
        "random_gauge_diag_pm1_max_error_float64": gauge_err_f64,
        "random_gauge_diag_pm1_max_error_float32": gauge_err_f32,
        "eta_solenoidal_grid": eta_grid,
        "imputation_by_model": imputation_curves,
        "homotopy_accuracy_mean": homotopy_mean,
        "homotopy_accuracy_std": homotopy_std,
        "linear_imputation_control": evaluate_linear_control(),
        "homotopy_seeds": seeds,
        "homotopy_accuracy_by_seed": homotopy_runs,
    }

    plot_snn_benchmarks(assets_dir, summary)
    return summary


def plot_snn_benchmarks(assets_dir: Path, summary: Dict[str, Any]) -> None:
    eta_grid = summary["eta_solenoidal_grid"]
    imputation_curves = summary["imputation_by_model"]
    homotopy_mean = summary["homotopy_accuracy_mean"]
    homotopy_std = summary["homotopy_accuracy_std"]
    model_names = list(imputation_curves)
    # High-contrast academic journal palette
    palette = {
        "NodeGCN": "#1B4F72",            # Navy
        "EdgeMLP": "#5D6D7E",            # Slate
        "Isotropic-L1-GCN": "#C55A11",   # Ochre
        "HodgeSimplicialNN": "#B03A2E",  # Crimson
    }
    markers = {
        "NodeGCN": "s",
        "EdgeMLP": "D",
        "Isotropic-L1-GCN": "^",
        "HodgeSimplicialNN": "o",
    }
    display_labels = {
        "NodeGCN": "NodeGCN ($\\mathbf{L}_0$ + Edge Diff)",
        "EdgeMLP": "EdgeMLP (Topology-Free)",
        "Isotropic-L1-GCN": "Isotropic $\\mathbf{L}_1$-GCN (Conflated)",
        "HodgeSimplicialNN": "HodgeSimplicialNN (Ours)",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.5), constrained_layout=True)
    _style_continuous_axes(ax1)
    _style_categorical_axes(ax2)

    pct_grid = [100.0 * e for e in eta_grid]
    for m in model_names:
        lw = 2.65 if m == "HodgeSimplicialNN" else 1.95
        ax1.plot(
            pct_grid,
            imputation_curves[m]["r2"],
            marker=markers[m],
            color=palette[m],
            markeredgecolor="#111111",
            markeredgewidth=0.60,
            linewidth=lw,
            markersize=6.6,
            zorder=5 if m == "HodgeSimplicialNN" else 4,
            label=display_labels[m],
        )

    linear_r2 = [row["missing_r2"] for row in summary["linear_imputation_control"]["rows"]]
    ax1.plot(pct_grid, linear_r2, color="#202020", ls=":", lw=2, marker="+",
             label="Training-flow subspace (linear)")
    ax1.axhline(0.0, color="#5D6D7E", linestyle="--", linewidth=1.05, zorder=2)
    # Placed cleanly in empty mid-upper space so pointer arrow crosses zero curves
    idx_75 = eta_grid.index(0.75)
    ax1.annotate(
        "NodeGCN output lies in gradient space\n($\\mathrm{im}(\\mathbf{B}_1^\\top) \\bot \\ker(\\mathbf{B}_1)$)",
        xy=(100.0, imputation_curves["NodeGCN"]["r2"][-1]),
        xytext=(38.0, 0.72),
        fontsize=8.4,
        color="#1B4F72",
        zorder=7,
        arrowprops=dict(arrowstyle="-|>", color="#1B4F72", lw=1.35, mutation_scale=11),
        bbox=dict(boxstyle="square,pad=0.28", facecolor="#FFFFFF", edgecolor="#1B4F72", lw=1.0, alpha=0.96),
    )

    ax1.set_xlabel("Solenoidal (Curl + Harmonic) Energy Share $\\eta_{\\mathrm{sol}}$ [%]")
    ax1.set_ylabel("Missing Edge Flux Reconstruction Test $R^2$")
    ax1.set_title("(a) 40% Missing Edge Flux Imputation vs. Flow Topology", pad=8)
    ax1.set_xticks(pct_grid)
    ax1.set_xlim(-4.0, 104.0)
    ax1.set_ylim(-0.12, 1.06)
    ax1.legend(
        loc="lower left",
        fancybox=False,
        edgecolor="#111111",
        framealpha=0.96,
        fontsize=8.2,
    )

    # Panel (b): Grouped comparison of Homotopy Classification Accuracy & Mixed-Flow (eta=0.75) R^2
    x_pos = np.arange(len(model_names))
    width = 0.36
    r2_mixed = [max(0.0, imputation_curves[m]["r2"][idx_75]) * 100.0 for m in model_names]
    acc_pct = [homotopy_mean[m] * 100.0 for m in model_names]
    acc_err = [homotopy_std[m] * 100.0 for m in model_names]

    bars1 = ax2.bar(
        x_pos - width / 2,
        acc_pct,
        width,
        yerr=acc_err,
        capsize=4,
        error_kw=dict(ecolor="#111111", elinewidth=1.2, capthick=1.2),
        color=[palette[m] for m in model_names],
        alpha=0.92,
        edgecolor="#111111",
        linewidth=0.95,
        zorder=3,
    )
    bars2 = ax2.bar(
        x_pos + width / 2,
        r2_mixed,
        width,
        color=[palette[m] for m in model_names],
        alpha=0.48,
        hatch="//",
        edgecolor="#111111",
        linewidth=0.95,
        zorder=3,
    )

    for b, err_v in zip(bars1, acc_err):
        h_val = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2,
            h_val + err_v + 1.8,
            f"{h_val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color="#111111",
        )
    for b in bars2:
        h_val = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2,
            h_val + 1.6,
            f"{h_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.1,
            color="#1B2631",
        )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        ["NodeGCN", "EdgeMLP", "Isotropic\n$\\mathbf{L}_1$-GCN", "HodgeSNN\n(Ours)"],
        fontweight="bold",
    )
    ax2.set_ylabel("Benchmark Score [% / $R^2 \\times 100$]")
    ax2.set_title("(b) Same-Mesh Radius Transfer & Imputation", pad=8)
    ax2.set_ylim(0.0, 117.0)

    legend_proxies = [
        Patch(
            facecolor="#4A5568",
            edgecolor="#111111",
            linewidth=0.95,
            label="Homotopy Loop Classification Acc. [%]",
        ),
        Patch(
            facecolor="#CBD5E1",
            edgecolor="#111111",
            linewidth=0.95,
            hatch="//",
            label="Mixed-Flow ($\\eta_{\\mathrm{sol}} = 75\\%$) $R^2 \\times 100$",
        ),
    ]
    centroid_accuracy = 100.0 * homotopy_mean["HarmonicCentroid"]
    reference = ax2.axhline(centroid_accuracy, color="#333333", ls=":", lw=1.3,
                            label=f"Harmonic centroid (classification): {centroid_accuracy:.1f}%")
    legend_proxies.append(reference)
    ax2.legend(
        handles=legend_proxies,
        loc="upper left",
        fancybox=False,
        edgecolor="#111111",
        framealpha=0.96,
        fontsize=8.2,
    )

    fig.suptitle(
        "Missing Edge Flows and Loop Classification on a Two-Hole Mesh",
        fontsize=12.5,
        fontweight="bold",
    )

    _finalize_journal_figure(fig)
    out_path = assets_dir / "hodge_snn_benchmark.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def generate_figure_4_spectrum(assets_dir: Path) -> Dict[str, Any]:
    """Generate 2-panel L1 Hodge Spectrum & Betti-1 Cavity-Filling Filtration figure."""
    sc = planar_punctured_complex(seed=42)
    filt = cavity_filling_filtration(sc, num_mesh_steps=26, num_lowest_eigs=16, seed=42)

    steps = filt["step"]
    beta_1 = filt["beta_1"]
    gap = filt["spectral_gap"]
    low_eigs = filt["low_eigenvalues"]
    mesh_end = int(filt["mesh_complete_step"])
    stage_labels = filt["stage_labels"]

    cap1_indices = np.where(stage_labels == "cap_hole_1")[0]
    cap2_indices = np.where(stage_labels == "cap_hole_2")[0]
    cap1_end = int(cap1_indices[-1]) if len(cap1_indices) > 0 else mesh_end
    cap2_end = int(cap2_indices[-1]) if len(cap2_indices) > 0 else cap1_end

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.5), constrained_layout=True)

    # Style dual-axis Panel (a) with 4-sided closed #111111 frame and inward ticks
    ax1.set_facecolor("#FFFFFF")
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#111111")
        spine.set_linewidth(1.30)
    ax1.tick_params(
        axis="x",
        which="major",
        direction="in",
        top=True,
        bottom=True,
        length=4.5,
        width=1.2,
        color="#111111",
        labelcolor="#111111",
    )
    ax1.tick_params(
        axis="y",
        which="major",
        direction="in",
        left=True,
        right=False,
        length=4.5,
        width=1.2,
        color="#B03A2E",
        labelcolor="#B03A2E",
    )
    ax1.grid(True, which="major", linestyle=":", linewidth=0.70, alpha=0.35, color="#94A3B8")

    # Highlight filtration phases
    ax1.axvspan(0, mesh_end, color="#F8FAFC", alpha=0.90, zorder=0, label="Phase I: Local Mesh Fill")
    ax1.axvspan(
        mesh_end, cap1_end, color="#FEF5E7", alpha=0.65, zorder=0,
        label="Phase IIa: Seal Cavity A ($\\beta_1: 2 \\to 1$)"
    )
    ax1.axvspan(
        cap1_end, cap2_end, color="#FDEDEC", alpha=0.65, zorder=0,
        label="Phase IIb: Seal Cavity B ($\\beta_1: 1 \\to 0$)"
    )

    ax1.plot(
        steps,
        beta_1,
        "o-",
        color="#B03A2E",
        markeredgecolor="#111111",
        markeredgewidth=0.35,
        linewidth=2.25,
        markersize=4.3,
        zorder=4,
        label="Betti Number $\\beta_1 = \\dim\\ker(\\mathbf{L}_1)$",
    )
    ax1.set_xlabel("2-Simplex Filtration & Cavity-Surgery Step $s$")
    ax1.set_ylabel("First Betti Number $\\beta_1$", color="#B03A2E", fontweight="bold")
    ax1.set_xlim(-1.5, cap2_end + 1.5)
    # Generous upper headroom so legend sits cleanly above both curves
    ax1.set_ylim(-6.0, 188.0)

    ax1_twin = ax1.twinx()
    for spine in ax1_twin.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#111111")
        spine.set_linewidth(1.30)
    ax1_twin.tick_params(
        axis="y",
        which="major",
        direction="in",
        right=True,
        left=False,
        length=4.5,
        width=1.2,
        color="#1B4F72",
        labelcolor="#1B4F72",
    )
    ax1_twin.plot(
        steps,
        gap,
        "s--",
        color="#1B4F72",
        markeredgecolor="#111111",
        markeredgewidth=0.35,
        linewidth=1.95,
        markersize=4.0,
        zorder=4,
        label="Spectral Gap $\\lambda_{\\Delta}(\\mathbf{L}_1) = \\lambda_{\\min}^+(\\mathbf{L}_1)$",
    )
    ax1_twin.set_ylabel(
        "1-Hodge Spectral Gap $\\lambda_{\\Delta}(\\mathbf{L}_1)$",
        color="#1B4F72",
        fontweight="bold",
    )
    ax1_twin.set_ylim(0.058, 0.148)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        fancybox=False,
        edgecolor="#111111",
        framealpha=0.96,
        fontsize=7.8,
    )
    ax1.set_title("(a) Betti-1 Decay, Topological Plateau & Cavity Annihilation", pad=8)

    # Panel (b): Lowest 16 eigenvalues of L1
    _style_continuous_axes(ax2)
    ax2.axvspan(mesh_end, cap1_end, color="#FEF5E7", alpha=0.55, zorder=0)
    ax2.axvspan(cap1_end, cap2_end, color="#FDEDEC", alpha=0.55, zorder=0)

    cmap_spec = plt.get_cmap("viridis")
    for k in range(2, low_eigs.shape[1]):
        ax2.plot(
            steps,
            low_eigs[:, k],
            color=cmap_spec(0.15 + 0.78 * (k / low_eigs.shape[1])),
            linewidth=1.25,
            alpha=0.80,
            zorder=2,
            label="Acoustic Modes ($k \\geq 3$)" if k == 2 else None,
        )

    # Highlight the 2 topological cavity zero-modes
    ax2.plot(
        steps,
        low_eigs[:, 1],
        color="#C55A11",
        linewidth=2.65,
        zorder=4,
        label="Cavity-A Mode $\\lambda_2(\\mathbf{L}_1)$",
    )
    ax2.plot(
        steps,
        low_eigs[:, 0],
        color="#B03A2E",
        linewidth=2.85,
        linestyle="-",
        zorder=5,
        label="Cavity-B Mode $\\lambda_1(\\mathbf{L}_1)$",
    )

    ax2.axvline(mesh_end, color="#5D6D7E", linestyle="--", linewidth=1.25, zorder=3)
    ax2.axvline(cap1_end, color="#C55A11", linestyle="--", linewidth=1.25, zorder=3)

    max_eig = float(np.max(low_eigs))
    ax2.text(
        mesh_end + 0.6,
        max_eig * 1.075,
        "Mesh Filled\n($\\beta_1 = 2$)",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#1B4F72",
        zorder=6,
        bbox=dict(boxstyle="square,pad=0.22", facecolor="#FFFFFF", edgecolor="#5D6D7E", lw=0.95, alpha=0.95),
    )
    ax2.text(
        cap1_end + 0.6,
        max_eig * 1.075,
        "Cavity A Sealed\n($\\beta_1 = 1$)",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#C55A11",
        zorder=6,
        bbox=dict(boxstyle="square,pad=0.22", facecolor="#FFFFFF", edgecolor="#C55A11", lw=0.95, alpha=0.95),
    )

    ax2.set_xlabel("2-Simplex Filtration & Cavity-Surgery Step $s$")
    ax2.set_ylabel("1-Hodge Eigenvalues $\\lambda_k(\\mathbf{L}_1)$")
    ax2.set_title("(b) Spectral Lifting of Harmonic Zero-Modes Under 2-Simplices", pad=8)
    ax2.set_xlim(-1.5, cap2_end + 1.5)
    ax2.set_ylim(-0.03, max_eig * 1.27)
    ax2.legend(
        loc="upper left",
        fancybox=False,
        edgecolor="#111111",
        framealpha=0.96,
        fontsize=8.0,
    )

    fig.suptitle(
        "Spectral Topology of 1-Hodge Laplacian $\\mathbf{L}_1 = \\mathbf{B}_1^\\top \\mathbf{B}_1 + \\mathbf{B}_2 \\mathbf{B}_2^\\top$: "
        "Mesh Filling vs. Cavity Surgery",
        fontsize=12.4,
        fontweight="bold",
    )

    _finalize_journal_figure(fig)
    out_path = assets_dir / "hodge_spectrum_cavity_filling.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "initial_1skeleton_beta_1": int(beta_1[0]),
        "punctured_mesh_beta_1": int(beta_1[mesh_end]),
        "after_cavity_a_cap_beta_1": int(beta_1[cap1_end]),
        "after_cavity_b_cap_beta_1": int(beta_1[cap2_end]),
        "punctured_spectral_gap": float(gap[mesh_end]),
        "fully_sealed_spectral_gap": float(gap[cap2_end]),
    }


def write_benchmark_table(results_dir: Path, snn_summary: Dict[str, Any]) -> pd.DataFrame:
    eta_grid = snn_summary["eta_solenoidal_grid"]
    idx_0 = eta_grid.index(0.0)
    idx_50 = eta_grid.index(0.5)
    idx_75 = eta_grid.index(0.75)
    idx_100 = eta_grid.index(1.0)

    rows: List[Dict[str, Any]] = []
    for m in ["NodeGCN", "EdgeMLP", "Isotropic-L1-GCN", "HodgeSimplicialNN"]:
        r2_list = snn_summary["imputation_by_model"][m]["r2"]
        mse_list = snn_summary["imputation_by_model"][m]["mse"]
        rows.append(
            {
                "Model": m,
                "Imputation_R2_GradOnly_eta0": round(r2_list[idx_0], 4),
                "Imputation_R2_Balanced_eta50": round(r2_list[idx_50], 4),
                "Imputation_R2_MixedSolenoidal_eta75": round(r2_list[idx_75], 4),
                "Imputation_R2_PureSolenoidal_eta100": round(r2_list[idx_100], 4),
                "Imputation_MSE_MixedSolenoidal_eta75": round(mse_list[idx_75], 4),
                "Homotopy_Acc_Mean": round(snn_summary["homotopy_accuracy_mean"][m], 4),
                "Homotopy_Acc_Std": round(snn_summary["homotopy_accuracy_std"][m], 4),
            }
        )

    rows.append({
        "Model": "HarmonicCentroid",
        "Homotopy_Acc_Mean": round(snn_summary["homotopy_accuracy_mean"]["HarmonicCentroid"], 4),
        "Homotopy_Acc_Std": round(snn_summary["homotopy_accuracy_std"]["HarmonicCentroid"], 4),
    })
    control = snn_summary["linear_imputation_control"]["rows"]
    rows.append({
        "Model": "TrainingFlowSubspace",
        "Imputation_R2_GradOnly_eta0": control[idx_0]["missing_r2"],
        "Imputation_R2_Balanced_eta50": control[idx_50]["missing_r2"],
        "Imputation_R2_MixedSolenoidal_eta75": control[idx_75]["missing_r2"],
        "Imputation_R2_PureSolenoidal_eta100": control[idx_100]["missing_r2"],
        "Imputation_MSE_MixedSolenoidal_eta75": control[idx_75]["missing_mse"],
    })
    df = pd.DataFrame(rows)
    csv_path = results_dir / "benchmark_table.csv"
    df.to_csv(csv_path, index=False)

    return df


def main() -> None:
    t0 = time.perf_counter()
    _setup_plot_style()

    assets_dir = ROOT / "assets"
    results_dir = ROOT / "results"
    assets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Generating Helmholtz-Hodge Decomposition figure & diagnostics...")
    hhd_summary = generate_figure_1_hhd(assets_dir)

    print("[2/4] Running Simplicial Contagion Hysteresis & Bistability sweeps...")
    contagion_summary = generate_figure_2_contagion(assets_dir)

    print("[3/4] Training & evaluating HodgeSimplicialNN and baselines...")
    snn_summary = generate_figure_3_snn_benchmarks(assets_dir)

    print("[4/4] Computing L1 Hodge Spectrum cavity-filling filtration...")
    spectrum_summary = generate_figure_4_spectrum(assets_dir)

    # Also summarize the trimolecular reaction complex
    chem_sc = trimolecular_reaction_complex(n_pollutants=6, seed=42)
    chem_b0, chem_b1 = compute_betti_numbers(chem_sc, method="svd")

    elapsed = time.perf_counter() - t0
    summary = {
        "project": "HigherOrder-Simplicial-Lab",
        "runtime_seconds": round(elapsed, 2),
        "helmholtz_hodge": hhd_summary,
        "simplicial_contagion": contagion_summary,
        "hodge_snn_benchmark": snn_summary,
        "hodge_spectrum_filtration": spectrum_summary,
        "trimolecular_reaction_complex": {
            "N0": chem_sc.num_nodes,
            "N1": chem_sc.num_edges,
            "N2": chem_sc.num_triangles,
            "beta_0": chem_b0,
            "beta_1": chem_b1,
            "boundary_identity_error": chem_sc.verify_boundary_identity(),
        },
    }

    json_path = results_dir / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Build clean tabular CSV summary across models
    df = write_benchmark_table(results_dir, snn_summary)

    print(f"\nAll benchmarks complete in {elapsed:.2f}s!")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
