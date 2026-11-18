"""Fit a linear subspace to the same complete training flows used by the neural models."""

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simplicial_lab import planar_punctured_complex
from simplicial_lab.models import compute_r2_and_mse, generate_edge_imputation_dataset


def evaluate_linear_control():
    mesh = planar_punctured_complex(seed=42)
    rows = []
    for eta in (0., .25, .5, .75, 1.):
        options = dict(mask_ratio=.40, solenoidal_fraction=eta)
        train = generate_edge_imputation_dataset(mesh, n_samples=150, seed=42, **options)
        test = generate_edge_imputation_dataset(mesh, n_samples=60, seed=10_042, **options)
        targets = train["y_true"].squeeze(-1).numpy()
        _, singular, vt = np.linalg.svd(targets.astype(np.float64), full_matrices=False)
        tolerance = np.finfo(targets.dtype).eps * max(targets.shape) * singular[0]
        basis = vt[singular > tolerance].T
        observed = test["x_obs"].squeeze(-1).numpy()
        masks = test["mask_obs"].squeeze(-1).numpy().astype(bool)
        pred = np.empty_like(observed)
        for i, mask in enumerate(masks):
            weights, *_ = np.linalg.lstsq(basis[mask], observed[i, mask], rcond=None)
            pred[i] = basis @ weights
        r2, mse = compute_r2_and_mse(torch.from_numpy(pred), test["y_true"], 1 - test["mask_obs"])
        rows.append({"solenoidal_fraction": eta, "train_rank": basis.shape[1],
                     "missing_r2": r2, "missing_mse": mse})
    result = {"train_seed": 42, "test_seed": 10_042, "train_count": 150, "test_count": 60,
              "masked_fraction": .40, "rows": rows}
    return result


def main():
    result = evaluate_linear_control()
    out = ROOT / "results/linear_imputation_control.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["rows"], indent=2))


if __name__ == "__main__":
    main()
