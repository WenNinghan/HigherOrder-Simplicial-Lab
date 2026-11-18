"""Fit on one mesh and score frozen parameters on three held-out two-hole meshes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simplicial_lab.complexes import planar_punctured_complex
from simplicial_lab.hodge import edge_winding_operator, harmonic_basis, hodge_laplacian_1
from simplicial_lab.models import (
    EdgeLineGraphGCN, HodgeSimplicialNN, SimplicialTrajectoryClassifier,
    generate_edge_imputation_dataset, generate_homotopy_trajectory_dataset,
)

MESHES = {
    "source": (13, 9, ((-.44, 0.), (.44, 0.)), .25, 42),
    "coarse": (12, 8, ((-.44, 0.), (.44, 0.)), .21, 1701),
    "shifted": (16, 11, ((-.48, .08), (.40, -.08)), .23, 1702),
    "fine": (18, 12, ((-.36, -.10), (.52, .10)), .26, 1703),
}
TRAIN_SEEDS = (42, 101, 202)
BACKBONES = ("edge_mlp", "node_gcn", "isotropic_l1", "hodge_snn")
MODE_COUNTS = (8, 16, 32, 64)
CLASS_WINDINGS = np.array([[1, 0], [-1, 0], [0, 1], [0, 0]])


def mesh_metadata(mesh, spec):
    n_components = connected_components(csr_matrix(abs(mesh.B1) @ abs(mesh.B1.T)),
                                        directed=False, return_labels=False)
    loops = mesh.metadata["hole_boundary_loops"]
    if n_components != 1 or mesh.euler_characteristic != -1 or len({frozenset(c) for c in loops}) != 2:
        raise ValueError("The task requires a connected planar mesh with two distinct inner boundaries.")
    digest = hashlib.sha256()
    for array in (mesh.coords, mesh.edges, mesh.triangles):
        digest.update(np.ascontiguousarray(array).tobytes())
    nx, ny, centers, radius, seed = spec
    return dict(nx=nx, ny=ny, hole_centers=centers, hole_radius=radius, mesh_seed=seed,
                n0_n1_n2=mesh.shape_tuple, mesh_sha256=digest.hexdigest())


def loop_data(mesh, seed, train):
    return generate_homotopy_trajectory_dataset(
        mesh, n_per_class=45 if train else 40, radius_range=(.27, .39) if train else (.43, .62),
        curl_noise_std=.45 if train else .55, grad_noise_std=.25 if train else .30, seed=seed,
    )


def flux_data(mesh, count, seed):
    return generate_edge_imputation_dataset(mesh, n_samples=count, mask_ratio=.4,
                                            solenoidal_fraction=.5, seed=seed)


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
    opt = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=1e-4)
    x, y = samples["x"].to(device), samples["y"].to(device)
    for _ in range(120):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    return model.eval()


def fit_imputer(mesh, samples, constructor, seed, device):
    torch.manual_seed(seed)
    model = constructor(mesh, hidden_channels=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=1e-5)
    x, y, mask = (samples[key].to(device) for key in ("x_obs", "y_true", "mask_obs"))
    missing = 1. - mask
    for _ in range(130):
        opt.zero_grad()
        sq = (model(x, mask=mask) - y).square()
        loss = .85 * (sq * missing).sum() / missing.sum() + .15 * (sq * mask).sum() / mask.sum()
        loss.backward()
        opt.step()
    return model.eval()


def harmonic_features(data, basis):
    return data["x"].squeeze(-1).numpy() @ basis


def nearest_centers(coords, centers):
    return ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def spectral_completion(data, basis):
    """Fit per-signal coefficients to the visible entries of a geometry-only dictionary."""
    obs = data["x_obs"].squeeze(-1).numpy()
    masks = data["mask_obs"].squeeze(-1).numpy().astype(bool)
    predicted = []
    for y, visible in zip(obs, masks):
        weights = np.linalg.lstsq(basis[visible], y[visible], rcond=1e-8)[0]
        predicted.append(basis @ weights)
    return np.array(predicted)


def flux_score(data, predicted):
    target = data["y_true"].squeeze(-1).numpy().astype(float)
    missing = data["mask_obs"].squeeze(-1).numpy() == 0
    errs = np.where(missing, target - predicted, 0.)
    sse = (errs ** 2).sum(axis=1)
    sst = np.sum((target[missing] - np.mean(target[missing])) ** 2)
    return dict(missing_r2=float(1. - sse.sum() / sst), missing_mse=float(sse.sum() / missing.sum()),
                missing_count=int(missing.sum()), missing_target_sst=float(sst), sse_by_signal=sse.tolist())


def fixed_generator_basis(mesh, harmonic):
    # Simulator-informed control. The generator chooses these specific low-frequency
    # vertex and dual-face modes. Their span is available from an unlabeled target mesh.
    val, vertex = np.linalg.eigh(mesh.B1 @ mesh.B1.T)
    grad = mesh.B1.T @ vertex[:, np.flatnonzero(val > 1e-6)[:10]]
    face_gram = mesh.B2.T @ mesh.B2
    np.fill_diagonal(face_gram, 0.)
    dual_adj = (face_gram != 0.).astype(float)
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
    low = {name: np.linalg.eigh(hodge_laplacian_1(mesh))[1][:, :max(MODE_COUNTS)] for name, mesh in meshes.items()}
    validation = flux_data(source, count=40, seed=35000)
    val_scores = {k: flux_score(validation, spectral_completion(validation, low["source"][:, :k]))["missing_mse"]
                  for k in MODE_COUNTS}
    k_selected = min(MODE_COUNTS, key=val_scores.get)
    tests, rows = {}, {}
    for i, (name, mesh) in enumerate(meshes.items()):
        loop_seed, flux_seed = 60000 + i, 70000 + i
        loops, flux = loop_data(mesh, loop_seed, train=False), flux_data(mesh, 60, flux_seed)
        tests[name] = (loops, flux)
        coords = harmonic_features(loops, harmonic[name])
        windings = coords @ (harmonic[name].T @ edge_winding_operator(mesh))
        labels = loops["y"].numpy()
        projected_winding_matches = np.max(np.abs(windings - CLASS_WINDINGS[labels]), axis=1) < 1e-4
        informed = fixed_generator_basis(mesh, harmonic[name])
        rows[name] = {
            **mesh_metadata(mesh, MESHES[name]), "loop_test_seed": loop_seed, "flux_test_seed": flux_seed,
            "class_labels": labels.tolist(), "winding_label_matches": int(projected_winding_matches.sum()),
            "loop_predictions_by_seed": {}, "flux_neural_by_seed": {},
            "flux_linear": {
                "low_hodge": flux_score(flux, spectral_completion(flux, low[name][:, :k_selected])),
                "generator_informed": flux_score(flux, spectral_completion(flux, informed)),
            },
        }
    for seed in TRAIN_SEEDS:
        loops = loop_data(source, seed, train=True)
        coords = harmonic_features(loops, harmonic["source"])
        centers = np.stack([coords[loops["y"].numpy() == k].mean(axis=0) for k in range(4)])
        for name in meshes:
            pred = nearest_centers(harmonic_features(tests[name][0], harmonic[name]), centers)
            rows[name]["loop_predictions_by_seed"][seed] = {"harmonic_centroid": pred.tolist()}
            rows[name]["flux_neural_by_seed"][seed] = {}
        for kind in BACKBONES:
            model = fit_classifier(source, loops, kind, seed, device)
            for name, mesh in meshes.items():
                target = copy_trainable(model, SimplicialTrajectoryClassifier(mesh, backbone_type=kind, hidden_channels=32).to(device))
                with torch.no_grad():
                    pred = target(tests[name][0]["x"].to(device)).argmax(dim=1)
                rows[name]["loop_predictions_by_seed"][seed][kind] = pred.cpu().tolist()
        flux = flux_data(source, 150, seed)
        for kind, constructor in (("hodge_snn", HodgeSimplicialNN), ("isotropic_l1", EdgeLineGraphGCN)):
            model = fit_imputer(source, flux, constructor, seed, device)
            for name, mesh in meshes.items():
                sample = tests[name][1]
                target = copy_trainable(model, constructor(mesh, hidden_channels=32).to(device))
                with torch.no_grad():
                    pred = target(sample["x_obs"].to(device), mask=sample["mask_obs"].to(device))
                rows[name]["flux_neural_by_seed"][seed][kind] = flux_score(sample, pred.squeeze(-1).cpu().numpy())
        print(f"Completed training seed {seed}", flush=True)
    return {
        "protocol": {
            "source_mesh": "source", "training_seeds": TRAIN_SEEDS, "test_samples_shared_across_training_seeds": True,
            "loop_train_per_class": 45, "loop_test_per_class": 40, "loop_epochs": 120,
            "flux_train_count": 150, "flux_test_count": 60, "flux_epochs": 130, "flux_mask": .4,
            "flux_solenoidal_fraction": .5, "source_validation_seed": 35000,
            "low_hodge_validation_mse": val_scores, "low_hodge_mode_count": k_selected,
            "generator_control": "Known generator mode family on target geometry; only observed values used for each solve.",
            "device": str(device), "torch": str(torch.__version__), "numpy": np.__version__,
        }, "meshes": rows,
    }


def plot_transfer(results, path):
    import matplotlib.pyplot as plt

    meshes = results["meshes"]
    neural = {"hodge_snn": "Hodge SNN", "isotropic_l1": "Isotropic L1"}
    loop_methods = {"harmonic_centroid": "Harmonic centroid", **neural,
                    "edge_mlp": "Edge MLP", "node_gcn": "Node GCN"}
    flux_methods = {**neural, "low_hodge": "Low Hodge modes", "generator_informed": "Generator-informed"}
    loop_matrix, flux_matrix = [], []
    for row in meshes.values():
        labels = np.asarray(row["class_labels"])
        loop_matrix.append([100. * np.mean([np.mean(np.asarray(p[k]) == labels)
                                             for p in row["loop_predictions_by_seed"].values()])
                            for k in loop_methods])
        flux_matrix.append([np.mean([r[k]["missing_r2"] for r in row["flux_neural_by_seed"].values()])
                            for k in neural] + [row["flux_linear"][k]["missing_r2"]
                                                for k in ("low_hodge", "generator_informed")])
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.15), layout="constrained")
    panels = [
        (np.array(loop_matrix).T, loop_methods, "Loop classification · accuracy (%)", 1, 0, 100),
        (np.array(flux_matrix).T, flux_methods, "Flow completion · missing-edge R²", 5,
         min(0, float(np.min(flux_matrix))), 1),
    ]
    for ax, (values, methods, title, decimals, vmin, vmax) in zip(axes, panels):
        heat = ax.imshow(values, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
        for (iy, ix), number in np.ndenumerate(values):
            ax.text(ix, iy, f"{number:.{decimals}f}", ha="center", va="center",
                    color="white" if heat.norm(number) > .58 else "#152f49")
        ax.set_xticks(range(len(meshes)), labels=[n.title() for n in meshes])
        ax.set_yticks(range(len(methods)), labels=list(methods.values()))
        ax.axvline(.5, color="#c4713e", linewidth=2)
        ax.set_title(title, fontsize=12)
        ax.tick_params(length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
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
