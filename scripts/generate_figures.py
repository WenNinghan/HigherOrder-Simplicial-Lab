"""Generate all five WRM print-native 600-DPI editorial figures for HigherOrder-Simplicial-Lab."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
import sys

if hasattr(platform, "_wmi"):
    platform._wmi = None

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import LineCollection, PolyCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simplicial_lab import (
    analytical_mean_field_equilibria,
    cavity_filling_filtration,
    edge_cochain_to_node_vectors,
    helmholtz_hodge_decomposition,
    iacopini_rsc_complex,
    planar_punctured_complex,
    run_hysteresis_sweep,
    simulate_simplicial_contagion_ode,
    simulate_stochastic_contagion,
    synthesize_hodge_flow,
)
from simplicial_lab.editorial_style import (
    C_BLUE,
    C_BRONZE,
    C_BURGUNDY,
    C_CHARCOAL,
    C_PLUM,
    C_SLATE,
    C_TEAL,
    HERO_WASH,
    INK,
    MM,
    PANEL_BG,
    REF_LINE,
    ROW_STRIPE,
    SAND_WASH,
    SEP_LINE,
    SPINE,
    add_panel_header,
    apply_editorial_rc,
    get_diverge_cmap,
    get_wine_cmap,
    save_editorial_png,
    style_panel_clean_heatmap,
    style_panel_horizontal,
    style_panel_tinted,
)

ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"


def _draw_mesh_flow(ax: plt.Axes, sc, flow: np.ndarray, arrow_col: str, tri_vals=None, node_vals=None) -> None:
    style_panel_clean_heatmap(ax)
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(SPINE)
        sp.set_linewidth(0.65)

    coords = sc.coords
    if sc.num_triangles > 0:
        verts = [coords[tri] for tri in sc.triangles]
        if tri_vals is not None:
            vm = max(float(np.max(np.abs(tri_vals))), 1e-6)
            fc = get_diverge_cmap()(plt.Normalize(-vm, vm)(tri_vals))
            fc[:, 3] = 0.55
            ax.add_collection(PolyCollection(verts, facecolors=fc, edgecolors="none", zorder=1))
        else:
            ax.add_collection(PolyCollection(verts, facecolors="#E5EBEF", alpha=0.65, edgecolors="none", zorder=1))

    segs = np.stack([coords[sc.edges[:, 0]], coords[sc.edges[:, 1]]], axis=1)
    af = np.abs(flow)
    fmax = max(float(np.percentile(af, 98)), 1e-6)
    st = np.clip(af / fmax, 0.0, 1.0)
    cols = get_wine_cmap()(0.15 + 0.80 * st)
    cols[:, 3] = 0.32 + 0.62 * st
    ax.add_collection(LineCollection(segs, linewidths=0.45 + 1.65 * st, colors=cols, zorder=2))

    if node_vals is not None:
        vmn = max(float(np.max(np.abs(node_vals))), 1e-6)
        ax.scatter(coords[:, 0], coords[:, 1], c=node_vals, cmap=get_diverge_cmap(), vmin=-vmn, vmax=vmn, s=14, edgecolors=INK, linewidths=0.35, zorder=4)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], color=INK, s=6, alpha=0.55, zorder=3)

    nvecs = edge_cochain_to_node_vectors(sc, flow)
    vn = np.linalg.norm(nvecs, axis=1)
    vref = max(float(np.percentile(vn, 95)), 1e-8)
    svecs = (nvecs / vref) * 0.12
    keep = vn > 0.10 * vref
    ax.quiver(
        coords[keep, 0], coords[keep, 1], svecs[keep, 0], svecs[keep, 1],
        angles="xy", scale_units="xy", scale=1.0, color=arrow_col,
        width=0.0048, headwidth=3.6, headlength=4.2, alpha=0.90, clip_on=True, zorder=5,
    )

    hole_r = sc.metadata.get("hole_radius", 0.24) * 0.72
    for idx, (cx, cy) in enumerate(sc.metadata.get("hole_centers", [])):
        ax.add_patch(mpatches.Circle((cx, cy), radius=hole_r, facecolor="white", edgecolor=C_BURGUNDY, ls="--", lw=1.0, zorder=6))
        ax.text(cx, cy, "A" if idx == 0 else "B", ha="center", va="center", fontsize=9.2, color=C_BURGUNDY, zorder=7)

    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-0.84, 0.84)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def make_fig1_hhd() -> None:
    apply_editorial_rc()
    sc = planar_punctured_complex(seed=42)
    synth = synthesize_hodge_flow(sc, grad_weight=1.15, curl_weight=1.00, harm_weight=1.05, seed=42)
    hhd = helmholtz_hodge_decomposition(sc, synth["f_total"])

    fig = plt.figure(figsize=(194 * MM, 126 * MM), dpi=600)
    gs = fig.add_gridspec(2, 2, left=0.02, right=0.98, bottom=0.03, top=0.92, wspace=0.08, hspace=0.22)

    panels = [
        (0, 0, "a", f"Total edge flow ($||f||_2^2 = {hhd.energies['total']:.2f}$)", hhd.f_total, C_CHARCOAL, None, None),
        (0, 1, "b", f"Gradient $B_1^\\top\\phi$ ({100*hhd.energy_fractions['grad']:.1f}%)", hhd.f_grad, C_BLUE, None, hhd.phi - np.mean(hhd.phi)),
        (1, 0, "c", f"Curl $B_2\\psi$ ({100*hhd.energy_fractions['curl']:.1f}%)", hhd.f_curl, C_PLUM, hhd.psi, None),
        (1, 1, "d", f"Harmonic $\\ker(L_1)$ ({100*hhd.energy_fractions['harm']:.1f}%)", hhd.f_harm, C_TEAL, None, None),
    ]
    for r, c, tag, title, fl, acol, tv, nv in panels:
        ax = fig.add_subplot(gs[r, c])
        _draw_mesh_flow(ax, sc, fl, acol, tri_vals=tv, node_vals=nv)
        add_panel_header(ax, tag, title, x_title=0.10, y=1.03)

    save_editorial_png(fig, ASSETS / "helmholtz_hodge_decomposition.png")


def make_fig2_snn_and_anisotropic() -> None:
    apply_editorial_rc()
    bench = json.loads((RESULTS / "benchmark_summary.json").read_text(encoding="utf-8"))["hodge_snn_benchmark"]
    lin_ctrl = json.loads((RESULTS / "linear_imputation_control.json").read_text(encoding="utf-8"))
    aniso_ctrl = json.loads((RESULTS / "anisotropic_constitutive_control.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(194 * MM, 88 * MM), dpi=600)
    gs = fig.add_gridspec(1, 2, left=0.072, right=0.988, bottom=0.15, top=0.845, wspace=0.66, width_ratios=[0.95, 1.08])

    # Panel (a): Same-mesh 40% missing-edge R^2 vs solenoidal share (with SVD linear control)
    ax0 = fig.add_subplot(gs[0, 0])
    style_panel_tinted(ax0, grid_axis="both")
    pct = [100.0 * e for e in bench["eta_solenoidal_grid"]]
    lin_r2 = [r["missing_r2"] for r in lin_ctrl["rows"]]
    ax0.plot(pct, lin_r2, color=C_CHARCOAL, ls="--", lw=1.6, marker="+", markersize=6, label="Training-flow SVD", zorder=5)
    for key, lbl, col, mk in [
        ("HodgeSimplicialNN", "HodgeSNN (decoupled)", C_BURGUNDY, "o"),
        ("NodeGCN", "NodeGCN ($L_0$ grad)", C_BLUE, "s"),
        ("Isotropic-L1-GCN", "Unsigned-LG/L1", C_BRONZE, "^"),
        ("EdgeMLP", "EdgeMLP", C_SLATE, "D"),
    ]:
        ax0.plot(pct, bench["imputation_by_model"][key]["r2"], color=col, lw=1.7, marker=mk, markersize=4.8, label=lbl, zorder=4)

    ax0.set_xlim(-4, 104)
    ax0.set_ylim(-0.08, 1.07)
    ax0.set_xticks([0, 25, 50, 75, 100])
    ax0.set_xlabel("Solenoidal energy share $\\eta_{\\mathrm{sol}}$ (%)")
    ax0.set_ylabel("Missing-edge test $R^2$")
    ax0.legend(loc="center left", bbox_to_anchor=(0.02, 0.27), frameon=False, fontsize=9.0, handletextpad=0.35, labelspacing=0.20)
    add_panel_header(ax0, "a", "Low-mode imputation (244 edges)", x_title=0.12, y=1.04)

    # Panel (b): 10-seed parameter-matched Tied-L1 vs HodgeSNN (Isotropic vs Anisotropic on Unseen 398 mesh)
    ax1 = fig.add_subplot(gs[0, 1])
    style_panel_horizontal(ax1)
    iso_u = aniso_ctrl["isotropic_imputation_control"]["unseen_shifted_398"]
    ani_u = aniso_ctrl["anisotropic_constitutive_operator"]["unseen_shifted_398"]

    rows = [
        ("Iso. unseen: HodgeSNN", iso_u["hodge_snn"], C_BURGUNDY, HERO_WASH),
        ("Iso. unseen: Tied-$L_1$ SNN", iso_u["tied_l1_snn"], C_TEAL, None),
        ("Iso. unseen: Unsigned-LG", iso_u["unsigned_lg_gcn"], C_BRONZE, ROW_STRIPE),
        ("Aniso. unseen: HodgeSNN", ani_u["hodge_snn"], C_BURGUNDY, HERO_WASH),
        ("Aniso. unseen: Tied-$L_1$ SNN", ani_u["tied_l1_snn"], C_TEAL, None),
        ("Aniso. unseen: Unsigned-LG", ani_u["unsigned_lg_gcn"], C_BRONZE, ROW_STRIPE),
    ]
    yp = np.arange(len(rows))[::-1]
    for yi, (lbl, vals, col, wash) in zip(yp, rows):
        if wash is not None:
            ax1.axhspan(yi - 0.43, yi + 0.43, color=wash, zorder=0)
        arr = np.asarray(vals, dtype=float)
        mv = float(arr.mean())
        sd = float(arr.std(ddof=1))
        ax1.plot([mv - sd, mv + sd], [yi, yi], color=col, lw=2.0, zorder=3)
        ax1.scatter([mv], [yi], color=col, s=34, zorder=4)
        tx = mv + sd + 0.025 if mv + sd < 0.86 else mv - sd - 0.025
        tha = "left" if mv + sd < 0.86 else "right"
        ty = yi if mv + sd < 0.86 else yi + 0.24
        ax1.text(tx, ty, f"{mv:.3f}", va="center", ha=tha, fontsize=9.0, color=INK)

    ax1.axhline(2.5, color=SEP_LINE, lw=0.85, ls="--", zorder=2)
    ax1.set_yticks(yp)
    ax1.set_yticklabels([r[0] for r in rows])
    ax1.set_xlim(-0.08, 1.04)
    ax1.set_ylim(-0.6, len(rows) - 0.4)
    ax1.set_xlabel("Unseen 398-edge test $R^2$ (10 seeds, mean $\\pm$ SD)")
    add_panel_header(ax1, "b", "Matched Tied-$L_1$ control", x_title=0.11, y=1.04)

    save_editorial_png(fig, ASSETS / "hodge_snn_benchmark.png")


def make_fig3_mesh_transfer() -> None:
    apply_editorial_rc()
    mt = json.loads((RESULTS / "mesh_transfer.json").read_text(encoding="utf-8"))
    meshes = [("source", "Src\n244"), ("coarse", "Crs\n212"), ("shifted", "Shf\n398"), ("fine", "Fine\n474")]
    xs = np.arange(len(meshes))

    fig = plt.figure(figsize=(194 * MM, 84 * MM), dpi=600)
    gs = fig.add_gridspec(1, 3, left=0.068, right=0.988, bottom=0.17, top=0.77, wspace=0.27)

    # Panel (a): Loop classification accuracy (%)
    ax0 = fig.add_subplot(gs[0, 0])
    style_panel_tinted(ax0, grid_axis="both")
    for mkey, lbl, col, mk, ls in [
        ("harmonic_centroid", "Harmonic centroid", C_CHARCOAL, "*", "--"),
        ("hodge_snn", "HodgeSNN", C_BURGUNDY, "o", "-"),
        ("isotropic_l1", "Unsigned-LG/L1", C_BRONZE, "^", "-"),
        ("node_gcn", "NodeGCN", C_BLUE, "s", ":"),
    ]:
        means = []
        for mk_name, _ in meshes:
            mr = mt["meshes"][mk_name]
            yt = np.asarray(mr["class_labels"], dtype=int)
            accs = [100.0 * np.mean(np.asarray(mr["loop_predictions_by_seed"][s][mkey]) == yt) for s in ("42", "101", "202")]
            means.append(float(np.mean(accs)))
        ax0.plot(xs, means, color=col, marker=mk, ls=ls, lw=1.7, markersize=5.4, label=lbl, zorder=4)
    ax0.axvline(0.5, color=REF_LINE, lw=0.9, ls="--", zorder=2)
    ax0.set_xticks(xs)
    ax0.set_xticklabels([m[1] for m in meshes])
    ax0.set_ylim(15, 107)
    ax0.set_ylabel("Loop winding accuracy (%)")
    add_panel_header(ax0, "a", "Loop classes (480 unseen)", x_title=0.14, y=1.04)

    # Panel (b): Smooth 22-mode flow R^2
    ax1 = fig.add_subplot(gs[0, 1])
    style_panel_tinted(ax1, grid_axis="both")
    gen_r2 = [mt["meshes"][m]["flux_linear"]["generator_informed"]["missing_r2"] for m, _ in meshes]
    low_r2 = [mt["meshes"][m]["flux_linear"]["low_hodge"]["missing_r2"] for m, _ in meshes]
    ax1.plot(xs, gen_r2, color=C_CHARCOAL, ls="--", lw=1.6, marker="+", markersize=6.5, label="Known 22-mode basis", zorder=5)
    ax1.plot(xs, low_r2, color=C_TEAL, ls="-.", lw=1.6, marker="D", markersize=4.6, label="32 low Hodge modes", zorder=4)
    for mkey, col, mk in [("hodge_snn", C_BURGUNDY, "o"), ("isotropic_l1", C_BRONZE, "^")]:
        vals = [float(np.mean([mt["meshes"][m]["flux_neural_by_seed"][s][mkey]["missing_r2"] for s in ("42", "101", "202")])) for m, _ in meshes]
        ax1.plot(xs, vals, color=col, marker=mk, lw=1.7, markersize=5.0, zorder=4)
    ax1.axvline(0.5, color=REF_LINE, lw=0.9, ls="--", zorder=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([m[1] for m in meshes])
    ax1.set_ylim(0.35, 1.05)
    ax1.set_ylabel("Missing-edge $R^2$")
    add_panel_header(ax1, "b", "Smooth 22-mode flows", x_title=0.14, y=1.04)

    # Panel (c): Full-rank localized vortex R^2
    ax2 = fig.add_subplot(gs[0, 2])
    style_panel_tinted(ax2, grid_axis="both")
    vlow = [mt["meshes"][m]["localized_flux_linear"]["low_hodge"]["missing_r2"] for m, _ in meshes]
    ax2.plot(xs, vlow, color=C_TEAL, ls="-.", lw=1.6, marker="D", markersize=4.6, zorder=4)
    for mkey, col, mk in [("hodge_snn", C_BURGUNDY, "o"), ("isotropic_l1", C_BRONZE, "^"), ("node_gcn", C_BLUE, "s")]:
        vals = [float(np.mean([mt["meshes"][m]["localized_flux_neural_by_seed"][s][mkey]["missing_r2"] for s in ("42", "101", "202")])) for m, _ in meshes]
        ax2.plot(xs, vals, color=col, marker=mk, lw=1.7, markersize=5.0, zorder=4)
    ax2.axvline(0.5, color=REF_LINE, lw=0.9, ls="--", zorder=2)
    ax2.axhline(0.0, color=SEP_LINE, lw=0.8, ls=":", zorder=2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([m[1] for m in meshes])
    ax2.set_ylim(-0.10, 0.96)
    add_panel_header(ax2, "c", "Localized vortices (rank 162)", x_title=0.14, y=1.04)

    h0, l0 = ax0.get_legend_handles_labels()
    h1, l1 = ax1.get_legend_handles_labels()
    fig.legend(
        h0 + h1, l0 + l1, loc="upper center", bbox_to_anchor=(0.52, 0.985),
        ncol=3, frameon=False, fontsize=9.0, columnspacing=1.1, handletextpad=0.35,
    )
    save_editorial_png(fig, ASSETS / "mesh_transfer.png")


def make_fig4_contagion() -> None:
    apply_editorial_rc()
    from simplicial_lab.contagion import analytical_bifurcation_curve

    sc_rsc = iacopini_rsc_complex(n_nodes=180, mean_k1=18.0, mean_k2=6.0, seed=42)
    l1_grid = np.linspace(0.25, 1.45, 28)
    sw_bi = run_hysteresis_sweep(sc=sc_rsc, lambda_2=2.8, lambda_1_grid=l1_grid, t_relax=95.0, seed=42)
    sw_co = run_hysteresis_sweep(sc=sc_rsc, lambda_2=0.8, lambda_1_grid=l1_grid, t_relax=95.0, seed=42)

    fig = plt.figure(figsize=(194 * MM, 84 * MM), dpi=600)
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.985, bottom=0.155, top=0.855, wspace=0.29)

    # Panel (a): Hysteresis sweep
    ax0 = fig.add_subplot(gs[0, 0])
    style_panel_tinted(ax0, grid_axis="both")
    l1_dense = np.linspace(0.25, 1.45, 240)
    ana_bi = analytical_bifurcation_curve(l1_dense, lambda_2=2.8)
    tip_b = float(ana_bi["lambda_1_critical_backward"])

    ax0.axvspan(tip_b, 1.0, color=HERO_WASH, alpha=0.8, zorder=1)
    ax0.plot(l1_dense, ana_bi["upper_stable"], color=C_CHARCOAL, lw=1.4, ls="-", label="Mean-field $\\rho_+^*$", zorder=3)
    ax0.plot(l1_dense, ana_bi["middle_unstable"], color=REF_LINE, lw=1.2, ls="--", label="Separatrix $\\rho_-^*$", zorder=3)
    ax0.plot(l1_grid, sw_bi.rho_forward, "o-", color=C_BURGUNDY, markersize=3.8, lw=1.5, label="$\\lambda_2=2.8$ forward", zorder=5)
    ax0.plot(l1_grid, sw_bi.rho_backward, "s--", color=C_BRONZE, markersize=3.6, lw=1.5, label="$\\lambda_2=2.8$ backward", zorder=5)
    ax0.plot(l1_grid, sw_co.rho_forward, "^-", color=C_TEAL, markersize=3.6, lw=1.4, label="$\\lambda_2=0.8$ pairwise", zorder=4)

    ax0.set_xlim(0.25, 1.45)
    ax0.set_ylim(-0.03, 0.96)
    ax0.set_xlabel("Rescaled pairwise infectivity $\\lambda_1$")
    ax0.set_ylabel("Stationary active density $\\rho^*$")
    ax0.legend(loc="upper left", frameon=False, fontsize=9.0, handletextpad=0.35, labelspacing=0.20)
    add_panel_header(ax0, "a", "Bistable hysteresis ($N=180$)", x_title=0.11, y=1.04)

    # Panel (b): Basin splitting at lambda_1 = 0.78, lambda_2 = 2.8
    ax1 = fig.add_subplot(gs[0, 1])
    style_panel_tinted(ax1, grid_axis="both")
    eq_b = analytical_mean_field_equilibria(0.78, 2.8)
    r_sep = float(eq_b["unstable_separatrix"])
    r_end = float(eq_b["stable_endemic"])

    for idx, r0 in enumerate([0.04, 0.10, 0.19, 0.30, 0.50, 0.72]):
        tr = simulate_simplicial_contagion_ode(sc_rsc, lambda_1=0.78, lambda_2=2.8, rho_init=r0, t_max=40.0, dt=0.1, seed=100 + idx)
        col = C_BURGUNDY if r0 > r_sep else C_BLUE
        lbl = ("ODE $\\rho_0 > \\rho_-^*$" if idx == 5 else ("ODE $\\rho_0 < \\rho_-^*$" if idx == 0 else None))
        ax1.plot(tr["times"], tr["rho"], color=col, lw=1.55, alpha=0.88, label=lbl, zorder=4)

    st_h = simulate_stochastic_contagion(sc_rsc, lambda_1=0.78, lambda_2=2.8, rho_init=0.32, t_max=40.0, spontaneous_rate=0.0, seed=7)
    st_l = simulate_stochastic_contagion(sc_rsc, lambda_1=0.78, lambda_2=2.8, rho_init=0.10, t_max=40.0, spontaneous_rate=0.0, seed=8)
    ax1.plot(st_h["times"], st_h["rho"], color=C_BRONZE, lw=1.0, alpha=0.55, label="Markov realizations", zorder=3)
    ax1.plot(st_l["times"], st_l["rho"], color=C_SLATE, lw=1.0, alpha=0.55, zorder=3)
    ax1.axhline(r_sep, color=REF_LINE, ls="--", lw=1.2, label=f"$\\rho_-^* = {r_sep:.3f}$", zorder=3)
    ax1.axhline(r_end, color=C_CHARCOAL, ls="-.", lw=1.2, label=f"$\\rho_+^* = {r_end:.3f}$", zorder=3)

    ax1.set_xlim(0.0, 40.0)
    ax1.set_ylim(-0.03, 0.98)
    ax1.set_xlabel("Dimensionless time $\\tau = \\mu t$")
    ax1.set_ylabel("Active node fraction $\\rho(\\tau)$")
    ax1.legend(loc="upper right", ncol=2, frameon=False, fontsize=9.0, columnspacing=0.7, handletextpad=0.3, labelspacing=0.20)
    add_panel_header(ax1, "b", "Basin splitting ($\\lambda_1=0.78, \\lambda_2=2.8$)", x_title=0.11, y=1.04)

    save_editorial_png(fig, ASSETS / "simplicial_phase_transition.png")


def make_fig5_spectrum() -> None:
    apply_editorial_rc()
    sc = planar_punctured_complex(seed=42)
    filt = cavity_filling_filtration(sc, num_mesh_steps=26, num_lowest_eigs=16, seed=42)
    steps = filt["step"]
    b1 = filt["beta_1"]
    low_eigs = filt["low_eigenvalues"]
    m_end = int(filt["mesh_complete_step"])
    s_lbls = filt["stage_labels"]
    c1_end = int(np.where(s_lbls == "cap_hole_1")[0][-1])
    c2_end = int(np.where(s_lbls == "cap_hole_2")[0][-1])

    fig = plt.figure(figsize=(194 * MM, 84 * MM), dpi=600)
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.985, bottom=0.155, top=0.855, wspace=0.29)

    # Panel (a): Betti-1 trajectory across filtration & cavity surgery
    ax0 = fig.add_subplot(gs[0, 0])
    style_panel_tinted(ax0, grid_axis="both")
    ax0.axvspan(m_end, c1_end, color=SAND_WASH, alpha=0.95, zorder=1, label="Cap Hole A ($2\\to 1$)")
    ax0.axvspan(c1_end, c2_end, color=HERO_WASH, alpha=0.95, zorder=1, label="Cap Hole B ($1\\to 0$)")
    ax0.plot(steps, b1, "o-", color=C_BURGUNDY, markersize=3.5, lw=1.7, label="$\\beta_1 = \\dim\\ker(L_1)$", zorder=4)
    ax0.set_xlim(-1.0, c2_end + 1.0)
    ax0.set_ylim(-6.0, 165.0)
    ax0.set_xlabel("2-simplex filtration step $s$")
    ax0.set_ylabel("First Betti number $\\beta_1$")
    ax0.legend(loc="upper right", frameon=False, fontsize=9.0, handletextpad=0.35, labelspacing=0.22)
    add_panel_header(ax0, "a", "Betti-1 decay and cavity surgery", x_title=0.11, y=1.04)

    # Panel (b): Lowest 16 eigenvalues of L1
    ax1 = fig.add_subplot(gs[0, 1])
    style_panel_tinted(ax1, grid_axis="both")
    ax1.axvspan(m_end, c1_end, color=SAND_WASH, alpha=0.95, zorder=1)
    ax1.axvspan(c1_end, c2_end, color=HERO_WASH, alpha=0.95, zorder=1)
    for k in range(2, low_eigs.shape[1]):
        ax1.plot(steps, low_eigs[:, k], color=C_SLATE, lw=0.95, alpha=0.65, zorder=2, label="Non-harmonic ($k\\geq 3$)" if k == 2 else None)
    ax1.plot(steps, low_eigs[:, 1], color=C_TEAL, lw=2.0, zorder=4, label="Hole A mode $\\lambda_2(L_1)$")
    ax1.plot(steps, low_eigs[:, 0], color=C_BURGUNDY, lw=2.1, zorder=5, label="Hole B mode $\\lambda_1(L_1)$")
    ax1.axvline(m_end, color=REF_LINE, ls="--", lw=0.95, zorder=3)
    ax1.axvline(c1_end, color=REF_LINE, ls=":", lw=0.95, zorder=3)

    ymax = float(np.max(low_eigs)) * 1.24
    ax1.set_xlim(-1.0, c2_end + 1.0)
    ax1.set_ylim(-0.02, ymax)
    ax1.set_xlabel("2-simplex filtration step $s$")
    ax1.set_ylabel("Eigenvalues $\\lambda_k(L_1)$")
    ax1.legend(loc="upper left", frameon=False, fontsize=9.0, handletextpad=0.35, labelspacing=0.22)
    add_panel_header(ax1, "b", "Spectral lifting of zero modes", x_title=0.11, y=1.04)

    save_editorial_png(fig, ASSETS / "hodge_spectrum_cavity_filling.png")


def main() -> None:
    make_fig1_hhd()
    make_fig2_snn_and_anisotropic()
    make_fig3_mesh_transfer()
    make_fig4_contagion()
    make_fig5_spectrum()


if __name__ == "__main__":
    main()
