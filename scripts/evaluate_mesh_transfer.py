"""Fit on one mesh and score frozen parameters on three held-out two-hole meshes.

Evaluates:
  1. Topological loop winding classification across 4 signed homotopy classes;
  2. Low-frequency spectral flow completion (with honest known-generator linear control);
  3. Full-rank spatially localized vortex + dipole flow completion (effective rank > 160,
     where random-center local eddies & saturating edge permeability break low-rank linear
     subspace assumptions, isolating local Hodge boundary/coboundary message passing).
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simplicial_lab.complexes import planar_punctured_complex
from simplicial_lab.hodge import edge_winding_operator, harmonic_basis, hodge_laplacian_1
from simplicial_lab.models import (
    EdgeLineGraphGCN,
    HodgeSimplicialNN,
    NodeGCNEdgePredictor,
    SimplicialTrajectoryClassifier,
    generate_edge_imputation_dataset,
    generate_homotopy_trajectory_dataset,
)

MESHES = {
    "source": (13, 9, ((-0.44, 0.0), (0.44, 0.0)), 0.25, 42),
    "coarse": (12, 8, ((-0.44, 0.0), (0.44, 0.0)), 0.21, 1701),
    "shifted": (16, 11, ((-0.48, 0.08), (0.40, -0.08)), 0.23, 1702),
    "fine": (18, 12, ((-0.36, -0.10), (0.52, 0.10)), 0.26, 1703),
}
TRAIN_SEEDS = (42, 101, 202)
BACKBONES = ("edge_mlp", "node_gcn", "isotropic_l1", "hodge_snn")
MODE_COUNTS = (8, 16, 32, 64)
CLASS_WINDINGS = np.array([[1, 0], [-1, 0], [0, 1], [0, 0]])


def mesh_metadata(mesh, spec):
    n_components = connected_components(
        csr_matrix(abs(mesh.B1) @ abs(mesh.B1.T)), directed=False, return_labels=False
    )
    loops = mesh.metadata["hole_boundary_loops"]
    if n_components != 1 or mesh.euler_characteristic != -1 or len({frozenset(c) for c in loops}) != 2:
        raise ValueError("The task requires a connected planar mesh with two distinct inner boundaries.")
    digest = hashlib.sha256()
    for array in (mesh.coords, mesh.edges, mesh.triangles):
        digest.update(np.ascontiguousarray(array).tobytes())
    nx, ny, centers, radius, seed = spec
    return dict(
        nx=nx,
        ny=ny,
        hole_centers=centers,
        hole_radius=radius,
        mesh_seed=seed,
        n0_n1_n2=mesh.shape_tuple,
        mesh_sha256=digest.hexdigest(),
    )


def loop_data(mesh, seed, train):
    return generate_homotopy_trajectory_dataset(
        mesh,
        n_per_class=45 if train else 40,
        radius_range=(0.27, 0.39) if train else (0.43, 0.62),
        curl_noise_std=0.45 if train else 0.55,
        grad_noise_std=0.25 if train else 0.30,
        seed=seed,
    )


def flux_data(mesh, count, seed):
    return generate_edge_imputation_dataset(
        mesh, n_samples=count, mask_ratio=0.4, solenoidal_fraction=0.5, seed=seed
    )


def localized_vortex_flux_data(mesh, count, seed, mask_ratio=0.40, patch_sigma=0.16, noise_std=0.03):
    """Synthesize full-rank spatially localized vortex + dipole + harmonic edge cochains."""
    rng = np.random.default_rng(seed)
    B1 = mesh.B1.astype(float)
    B2 = mesh.B2.astype(float)
    U_h = harmonic_basis(mesh)
    v_xy = mesh.coords
    t_xy = v_xy[mesh.triangles].mean(axis=1)
    ys, x_obs, masks = [], [], []
    for _ in range(count):
        psi = np.zeros(len(t_xy), dtype=float)
        for _ in range(4):
            c = t_xy[rng.integers(len(t_xy))]
            psi += rng.normal(0.0, 1.0) * np.exp(-np.sum((t_xy - c) ** 2, axis=1) / (2.0 * patch_sigma**2))
        phi = np.zeros(len(v_xy), dtype=float)
        for _ in range(4):
            c = v_xy[rng.integers(len(v_xy))]
            phi += rng.normal(0.0, 0.8) * np.exp(-np.sum((v_xy - c) ** 2, axis=1) / (2.0 * patch_sigma**2))
        f_lin = B2 @ psi + B1.T @ phi + U_h @ rng.normal(0.0, 0.9, size=U_h.shape[1])
        f_lin = f_lin / (np.std(f_lin) + 1e-8)
        # Odd saturating permeability preserves exact orientation equivariance f(-x) = -f(x)
        f = np.tanh(1.35 * f_lin) + 0.45 * f_lin
        f = f / (np.std(f) + 1e-8)
        n_e = len(f)
        m = np.ones(n_e, dtype=np.float32)
        m[rng.choice(n_e, size=int(round(mask_ratio * n_e)), replace=False)] = 0.0
        noisy = f + rng.normal(0.0, noise_std, size=n_e)
        ys.append(f.astype(np.float32))
        x_obs.append((noisy * m).astype(np.float32))
        masks.append(m)
    return {
        "x_obs": torch.tensor(np.array(x_obs))[..., None],
        "y_true": torch.tensor(np.array(ys))[..., None],
        "mask_obs": torch.tensor(np.array(masks))[..., None],
    }


def copy_trainable(source, target):
    """Transfer learned weights only; target boundary, harmonic and Laplacian buffers stay local."""
    src = dict(source.named_parameters())
    dst = dict(target.named_parameters())
    if src.keys() != dst.keys():
        raise ValueError("Models have incompatible learned parameters.")
    with torch.no_grad():
        for name, parameter in dst.items():
            parameter.copy_(src[name])
    return target.eval()


def fit_classifier(mesh, samples, kind, seed, device):
    torch.manual_seed(seed)
    model = SimplicialTrajectoryClassifier(mesh, backbone_type=kind, hidden_channels=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    x, y = samples["x"].to(device), samples["y"].to(device)
    for _ in range(120):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    return model.eval()


def fit_imputer(mesh, samples, constructor, seed, device, epochs=130):
    torch.manual_seed(seed)
    model = constructor(mesh, hidden_channels=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    x, y, mask = (samples[key].to(device) for key in ("x_obs", "y_true", "mask_obs"))
    missing = 1.0 - mask
    for _ in range(epochs):
        opt.zero_grad()
        sq = (model(x, mask=mask) - y).square()
        loss = 0.85 * (sq * missing).sum() / missing.sum() + 0.15 * (sq * mask).sum() / mask.sum()
        loss.backward()
        opt.step()
    return model.eval()


def harmonic_features(data, basis):
    return data["x"].squeeze(-1).numpy() @ basis


def nearest_centers(coords, centers):
    return ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def spectral_completion(data, basis, rcond=1e-8):
    """Fit per-signal coefficients to the visible entries of a geometry-only dictionary."""
    obs = data["x_obs"].squeeze(-1).numpy()
    masks = data["mask_obs"].squeeze(-1).numpy().astype(bool)
    predicted = []
    for y, visible in zip(obs, masks):
        weights = np.linalg.lstsq(basis[visible], y[visible], rcond=rcond)[0]
        predicted.append(basis @ weights)
    return np.array(predicted)


def flux_score(data, predicted):
    target = data["y_true"].squeeze(-1).numpy().astype(float)
    missing = data["mask_obs"].squeeze(-1).numpy() == 0
    errs = np.where(missing, target - predicted, 0.0)
    sse = (errs**2).sum(axis=1)
    sst = np.sum((target[missing] - np.mean(target[missing])) ** 2)
    return dict(
        missing_r2=float(1.0 - sse.sum() / sst),
        missing_mse=float(sse.sum() / missing.sum()),
        missing_count=int(missing.sum()),
        missing_target_sst=float(sst),
        sse_by_signal=sse.tolist(),
    )


def fixed_generator_basis(mesh, harmonic):
    val, vertex = np.linalg.eigh(mesh.B1 @ mesh.B1.T)
    grad = mesh.B1.T @ vertex[:, np.flatnonzero(val > 1e-6)[:10]]
    face_gram = mesh.B2.T @ mesh.B2
    np.fill_diagonal(face_gram, 0.0)
    dual_adj = (face_gram != 0.0).astype(float)
    _, face = np.linalg.eigh(np.diag(dual_adj.sum(axis=1)) - dual_adj)
    modes = np.column_stack((grad, mesh.B2 @ face[:, :10], harmonic))
    left, singular, _ = np.linalg.svd(modes, full_matrices=False)
    return left[:, singular > 1e-8 * singular[0]]


def run(device):
    meshes = {name: planar_punctured_complex(*spec) for name, spec in MESHES.items()}
    source = meshes["source"]
    harmonic = {name: harmonic_basis(mesh) for name, mesh in meshes.items()}
    if any(basis.shape[1] != 2 for basis in harmonic.values()):
        raise ValueError("Both named cavities must be present on every mesh.")
    low = {
        name: np.linalg.eigh(hodge_laplacian_1(mesh))[1][:, : max(MODE_COUNTS)]
        for name, mesh in meshes.items()
    }
    validation = flux_data(source, count=40, seed=35000)
    val_scores = {
        k: flux_score(validation, spectral_completion(validation, low["source"][:, :k]))["missing_mse"]
        for k in MODE_COUNTS
    }
    k_selected = min(MODE_COUNTS, key=val_scores.get)

    # Validation selection of spectral mode count for localized full-rank flow
    val_loc = localized_vortex_flux_data(source, count=40, seed=45000)
    val_loc_scores = {
        k: flux_score(val_loc, spectral_completion(val_loc, low["source"][:, :k], rcond=1e-4))["missing_mse"]
        for k in MODE_COUNTS
    }
    k_loc_selected = min(MODE_COUNTS, key=val_loc_scores.get)

    tests, rows = {}, {}
    for i, (name, mesh) in enumerate(meshes.items()):
        loop_seed, flux_seed, loc_seed = 60000 + i, 70000 + i, 80000 + i
        loops = loop_data(mesh, loop_seed, train=False)
        flux = flux_data(mesh, 60, flux_seed)
        loc_flux = localized_vortex_flux_data(mesh, 60, loc_seed)
        tests[name] = (loops, flux, loc_flux)
        coords = harmonic_features(loops, harmonic[name])
        windings = coords @ (harmonic[name].T @ edge_winding_operator(mesh))
        labels = loops["y"].numpy()
        projected_winding_matches = np.max(np.abs(windings - CLASS_WINDINGS[labels]), axis=1) < 1e-4
        informed = fixed_generator_basis(mesh, harmonic[name])
        rows[name] = {
            **mesh_metadata(mesh, MESHES[name]),
            "loop_test_seed": loop_seed,
            "flux_test_seed": flux_seed,
            "localized_flux_test_seed": loc_seed,
            "class_labels": labels.tolist(),
            "winding_label_matches": int(projected_winding_matches.sum()),
            "loop_predictions_by_seed": {},
            "flux_neural_by_seed": {},
            "localized_flux_neural_by_seed": {},
            "flux_linear": {
                "low_hodge": flux_score(flux, spectral_completion(flux, low[name][:, :k_selected])),
                "generator_informed": flux_score(flux, spectral_completion(flux, informed)),
            },
            "localized_flux_linear": {
                "low_hodge": flux_score(
                    loc_flux, spectral_completion(loc_flux, low[name][:, :k_loc_selected], rcond=1e-4)
                ),
            },
        }

    loc_eff_ranks = []
    source_svd_loc_scores = []
    for seed in TRAIN_SEEDS:
        loops = loop_data(source, seed, train=True)
        coords = harmonic_features(loops, harmonic["source"])
        centers = np.stack([coords[loops["y"].numpy() == k].mean(axis=0) for k in range(4)])
        for name in meshes:
            pred = nearest_centers(harmonic_features(tests[name][0], harmonic[name]), centers)
            rows[name]["loop_predictions_by_seed"][seed] = {"harmonic_centroid": pred.tolist()}
            rows[name]["flux_neural_by_seed"][seed] = {}
            rows[name]["localized_flux_neural_by_seed"][seed] = {}

        for kind in BACKBONES:
            model = fit_classifier(source, loops, kind, seed, device)
            for name, mesh in meshes.items():
                target = copy_trainable(
                    model, SimplicialTrajectoryClassifier(mesh, backbone_type=kind, hidden_channels=32).to(device)
                )
                with torch.no_grad():
                    pred = target(tests[name][0]["x"].to(device)).argmax(dim=1)
                rows[name]["loop_predictions_by_seed"][seed][kind] = pred.cpu().tolist()

        flux = flux_data(source, 150, seed)
        for kind, constructor in (("hodge_snn", HodgeSimplicialNN), ("isotropic_l1", EdgeLineGraphGCN)):
            model = fit_imputer(source, flux, constructor, seed, device, epochs=130)
            for name, mesh in meshes.items():
                sample = tests[name][1]
                target = copy_trainable(model, constructor(mesh, hidden_channels=32).to(device))
                with torch.no_grad():
                    pred = target(sample["x_obs"].to(device), mask=sample["mask_obs"].to(device))
                rows[name]["flux_neural_by_seed"][seed][kind] = flux_score(
                    sample, pred.squeeze(-1).cpu().numpy()
                )

        # Full-rank spatially localized vortex + dipole imputation & frozen cross-mesh transfer
        loc_train = localized_vortex_flux_data(source, 180, seed + 500)
        Y_loc = loc_train["y_true"].squeeze(-1).numpy()
        _, svals_loc, vt_loc = np.linalg.svd(Y_loc, full_matrices=False)
        loc_eff_ranks.append(int(np.sum(svals_loc > 1e-2 * svals_loc[0])))
        # Tune SVD rank on source validation set (cannot transfer to unseen meshes of different edge dimension)
        best_r = min((16, 32, 48, 64), key=lambda r: flux_score(
            val_loc, spectral_completion(val_loc, vt_loc[:r].T, rcond=1e-4)
        )["missing_mse"])
        source_svd_loc_scores.append(
            flux_score(tests["source"][2], spectral_completion(tests["source"][2], vt_loc[:best_r].T, rcond=1e-4))[
                "missing_r2"
            ]
        )

        for kind, constructor in (
            ("hodge_snn", HodgeSimplicialNN),
            ("isotropic_l1", EdgeLineGraphGCN),
            ("node_gcn", NodeGCNEdgePredictor),
        ):
            model = fit_imputer(source, loc_train, constructor, seed, device, epochs=150)
            for name, mesh in meshes.items():
                sample = tests[name][2]
                target = copy_trainable(model, constructor(mesh, hidden_channels=32).to(device))
                with torch.no_grad():
                    pred = target(sample["x_obs"].to(device), mask=sample["mask_obs"].to(device))
                rows[name]["localized_flux_neural_by_seed"][seed][kind] = flux_score(
                    sample, pred.squeeze(-1).cpu().numpy()
                )

        print(f"Completed training seed {seed}", flush=True)

    rows["source"]["localized_flux_linear"]["source_training_svd_mean_r2"] = float(np.mean(source_svd_loc_scores))
    return {
        "protocol": {
            "source_mesh": "source",
            "training_seeds": TRAIN_SEEDS,
            "test_samples_shared_across_training_seeds": True,
            "loop_train_per_class": 45,
            "loop_test_per_class": 40,
            "loop_epochs": 120,
            "flux_train_count": 150,
            "flux_test_count": 60,
            "flux_epochs": 130,
            "localized_flux_train_count": 180,
            "localized_flux_epochs": 150,
            "localized_flux_effective_ranks": loc_eff_ranks,
            "localized_source_training_svd_mean_r2": float(np.mean(source_svd_loc_scores)),
            "flux_mask": 0.4,
            "flux_solenoidal_fraction": 0.5,
            "source_validation_seed": 35000,
            "low_hodge_validation_mse": val_scores,
            "low_hodge_mode_count": k_selected,
            "localized_low_hodge_validation_mse": val_loc_scores,
            "localized_low_hodge_mode_count": k_loc_selected,
            "generator_control": "Known generator mode family on target geometry; only observed values used for each solve.",
            "device": str(device),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
        "meshes": rows,
    }


def plot_transfer(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "font.weight": "bold",
            "mathtext.fontset": "stix",
            "font.size": 10.2,
            "axes.titlesize": 11.4,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#111111",
            "axes.linewidth": 1.35,
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
        }
    )

    meshes = results["meshes"]
    mesh_labels = [
        f"Source\n({meshes['source']['n0_n1_n2'][1]}e)",
        f"Coarse\n({meshes['coarse']['n0_n1_n2'][1]}e)",
        f"Shifted\n({meshes['shifted']['n0_n1_n2'][1]}e)",
        f"Fine\n({meshes['fine']['n0_n1_n2'][1]}e)",
    ]

    loop_methods = {
        "harmonic_centroid": "Harmonic centroid",
        "hodge_snn": "Hodge SNN",
        "isotropic_l1": "Isotropic L1 GCN",
        "edge_mlp": "Edge MLP",
        "node_gcn": "Node GCN",
    }
    flux_methods = {
        "hodge_snn": "Hodge SNN",
        "isotropic_l1": "Isotropic L1 GCN",
        "low_hodge": "Low Hodge modes",
        "generator_informed": "Known 22-mode basis",
    }
    loc_methods = {
        "hodge_snn": "Hodge SNN",
        "isotropic_l1": "Isotropic L1 GCN",
        "node_gcn": "Node GCN",
        "low_hodge": "Low Hodge modes",
    }

    loop_matrix, flux_matrix, loc_matrix = [], [], []
    for row in meshes.values():
        labels = np.asarray(row["class_labels"])
        loop_matrix.append(
            [
                100.0 * np.mean([np.mean(np.asarray(p[k]) == labels) for p in row["loop_predictions_by_seed"].values()])
                for k in loop_methods
            ]
        )
        flux_matrix.append(
            [
                np.mean([r[k]["missing_r2"] for r in row["flux_neural_by_seed"].values()])
                for k in ("hodge_snn", "isotropic_l1")
            ]
            + [row["flux_linear"][k]["missing_r2"] for k in ("low_hodge", "generator_informed")]
        )
        loc_matrix.append(
            [
                np.mean([r[k]["missing_r2"] for r in row["localized_flux_neural_by_seed"].values()])
                for k in ("hodge_snn", "isotropic_l1", "node_gcn")
            ]
            + [row["localized_flux_linear"]["low_hodge"]["missing_r2"]]
        )

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.7))
    panels = [
        (np.array(loop_matrix).T, loop_methods, "(a) Loop Classification Accuracy (%)", "{:.1f}%", 20.0, 100.0, "YlGnBu"),
        (np.array(flux_matrix).T, flux_methods, r"(b) Smooth 22-Mode Flow $R^2$", "{:.4f}", 0.40, 1.00, "YlGnBu"),
        (np.array(loc_matrix).T, loc_methods, r"(c) Full-Rank Localized Vortex $R^2$", "{:.3f}", -0.25, 0.92, "YlGnBu"),
    ]

    for ax, (values, methods, title, fmt, vmin, vmax, cmap_name) in zip(axes, panels):
        heat = ax.imshow(values, cmap=cmap_name, vmin=vmin, vmax=vmax, aspect="auto")
        # Draw crisp interior cell gridlines
        for x_g in np.arange(-0.5, values.shape[1], 1.0):
            ax.axvline(x_g, color="#FFFFFF", linewidth=1.2, zorder=2)
        for y_g in np.arange(-0.5, values.shape[0], 1.0):
            ax.axhline(y_g, color="#FFFFFF", linewidth=1.2, zorder=2)
        # Highlight Source vs Unseen Meshes boundary
        ax.axvline(0.5, color="#B03A2E", linewidth=2.4, linestyle="--", zorder=4)

        for (iy, ix), number in np.ndenumerate(values):
            norm_val = heat.norm(np.clip(number, vmin, vmax))
            txt_col = "#FFFFFF" if norm_val > 0.56 else "#111111"
            ax.text(
                ix,
                iy,
                fmt.format(number),
                ha="center",
                va="center",
                fontsize=9.4,
                fontfamily="Times New Roman",
                fontweight="bold",
                color=txt_col,
                zorder=5,
            )

        ax.set_xticks(range(len(mesh_labels)), labels=mesh_labels)
        ax.set_yticks(range(len(methods)), labels=list(methods.values()))
        ax.set_title(title, pad=9, fontfamily="Times New Roman", fontweight="bold")
        ax.tick_params(axis="both", direction="out", length=3.8, width=1.2, colors="#111111")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#111111")
            spine.set_linewidth(1.35)
        for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            lbl.set_fontfamily("Times New Roman")
            lbl.set_fontweight("bold")

    fig.tight_layout(pad=1.15)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = run(torch.device(args.device))
    path = ROOT / "results/mesh_transfer.json"
    path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_transfer(results, ROOT / "assets/mesh_transfer.png")
    print(path)
