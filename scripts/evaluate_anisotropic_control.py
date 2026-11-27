"""Matched-architecture Signed Tied-L1 control vs. Decoupled HodgeSimplicialNN.

Addresses two methodological confounds in standard simplicial benchmarks:
1. EdgeLineGraphGCN mixes 75% unsigned line-graph adjacency with 25% L1 and lacks
   mask-observation gating, harmonic readout, and observed-edge skip connections.
   Here we introduce a parameter-matched, orientation-equivariant TiedL1SimplicialNN
   that shares every architectural component of HodgeSimplicialNN except replacing
   separate (L1_down^k, L1_up^k) filter weights with tied L1^k = (L1_down + L1_up)^k.
2. Evaluates 10 paired random seeds across source (244 edges) and unseen target
   meshes (398 edges) under:
   - Regime A (Isotropic imputation control): equal gradient/curl scaling where symmetric
     L1 message passing is well aligned with the data;
   - Regime B (Anisotropic divergence/curl constitutive operator): opposing irrotational
     vs. solenoidal transport where L1_down and L1_up share overlapping positive spectra
     so any polynomial p(L1) conflates gradient and curl modes;
   plus paired inference on the existing cross-mesh transfer results.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
import platform
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

if hasattr(platform, "_wmi"):
    platform._wmi = None

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.stats import binomtest, wilcoxon
import torch
import torch.nn as nn

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simplicial_lab.complexes import SimplicialComplex, planar_punctured_complex
from simplicial_lab.editorial_style import holm_adjust, paired_bootstrap_ci
from simplicial_lab.hodge import harmonic_basis
from simplicial_lab.models import (
    EdgeLineGraphGCN,
    HodgeSimplicialNN,
    _format_edge_input,
    _op_mul,
    _restore_edge_output,
    build_torch_operators,
)


class TiedL1ConvLayer(nn.Module):
    """Tied-L1 counterpart to HodgeSimplicialConvLayer using only L1_iso = norm(B1^T B1 + B2 B2^T)."""

    def __init__(self, in_channels: int, out_channels: int, filter_order: int = 2) -> None:
        super().__init__()
        self.filter_order = filter_order
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_mask_obs = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_iso = nn.ModuleList(
            [nn.Linear(in_channels, out_channels, bias=False) for _ in range(filter_order)]
        )

    def forward(self, h: torch.Tensor, l1_iso: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.lin_self(h)
        if mask is not None:
            out = out + self.lin_mask_obs(mask * h)
        h_k = h
        for k in range(self.filter_order):
            h_k = _op_mul(l1_iso, h_k)
            out = out + self.lin_iso[k](h_k)
        return out


class TiedL1SimplicialNN(nn.Module):
    """Orientation-equivariant architecture-matched control with tied L1 spectral filters.

    Matches HodgeSimplicialNN(hidden_channels=32) in parameter count when hidden_channels=40
    (26,000 vs. 25,792 parameters) while preserving exact mask gating, harmonic projection,
    and observed-edge residual skip structure.
    """

    def __init__(
        self,
        sc: SimplicialComplex,
        in_channels: int = 1,
        hidden_channels: int = 40,
        out_channels: int = 1,
        num_layers: int = 4,
        filter_order: int = 2,
    ) -> None:
        super().__init__()
        ops = build_torch_operators(sc, dtype=torch.float64)
        self.register_buffer("L1_iso_norm", ops["L1_iso_norm"].to(dtype=torch.float32))
        self.register_buffer("U_harm", ops["U_harm"].to(dtype=torch.float32))
        self.in_proj = nn.Linear(in_channels, hidden_channels, bias=False)
        self.layers = nn.ModuleList(
            [TiedL1ConvLayer(hidden_channels, hidden_channels, filter_order=filter_order) for _ in range(num_layers)]
        )
        self.harm_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.out_proj = nn.Linear(hidden_channels * 2, out_channels, bias=False)

    def rebind_mesh(self, sc: SimplicialComplex) -> "TiedL1SimplicialNN":
        clone = TiedL1SimplicialNN(
            sc=sc,
            in_channels=self.in_proj.in_features,
            hidden_channels=self.in_proj.out_features,
            out_channels=self.out_proj.out_features,
            num_layers=len(self.layers),
            filter_order=self.layers[0].filter_order,
        )
        src_params = dict(self.named_parameters())
        with torch.no_grad():
            for k, p in clone.named_parameters():
                p.copy_(src_params[k])
        return clone

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, residual_input: bool = True) -> torch.Tensor:
        x_3d, m_3d, orig_mode = _format_edge_input(x, mask)
        h = torch.tanh(self.in_proj(x_3d))
        h0 = h
        for layer in self.layers:
            h = h + torch.tanh(layer(h, self.L1_iso_norm, mask=m_3d))
            if m_3d is not None:
                h = h + 0.25 * (m_3d * h0)
        if self.U_harm.shape[1] > 0:
            hc = _op_mul(self.U_harm.transpose(0, 1).contiguous(), h)
            h_harm = _op_mul(self.U_harm, self.harm_proj(hc))
        else:
            h_harm = torch.zeros_like(h)
        out = self.out_proj(torch.cat([h, h_harm], dim=-1))
        if residual_input and m_3d is not None and out.shape[-1] == x_3d.shape[-1]:
            out = m_3d * x_3d + (1.0 - m_3d) * out + 0.15 * (m_3d * (out - x_3d))
        return _restore_edge_output(out, orig_mode)


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _generate_constitutive_dataset(
    mesh: SimplicialComplex,
    n_samples: int,
    seed: int,
    anisotropic: bool,
    mask_ratio: float = 0.40,
    patch_sigma: float = 0.16,
    noise_std: float = 0.03,
) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    B1 = mesh.B1.astype(float)
    B2 = mesh.B2.astype(float)
    U_h = harmonic_basis(mesh)
    v_xy = mesh.coords
    t_xy = v_xy[mesh.triangles].mean(axis=1)
    n_e = mesh.num_edges

    xs, ys, ms = [], [], []
    for _ in range(n_samples):
        psi = np.zeros(len(t_xy), dtype=float)
        for _ in range(4):
            c = t_xy[rng.integers(len(t_xy))]
            psi += rng.normal(0.0, 1.0) * np.exp(-np.sum((t_xy - c) ** 2, axis=1) / (2.0 * patch_sigma**2))
        phi = np.zeros(len(v_xy), dtype=float)
        for _ in range(4):
            c = v_xy[rng.integers(len(v_xy))]
            phi += rng.normal(0.0, 1.0) * np.exp(-np.sum((v_xy - c) ** 2, axis=1) / (2.0 * patch_sigma**2))

        f_curl = B2 @ psi
        f_grad = B1.T @ phi
        f_curl = f_curl / (np.std(f_curl) + 1e-8)
        f_grad = f_grad / (np.std(f_grad) + 1e-8)
        f_harm = U_h @ rng.normal(0.0, 0.7, size=U_h.shape[1])

        f_in = f_grad + f_curl + 0.5 * f_harm
        scale_in = np.std(f_in) + 1e-8
        f_in_norm = f_in / scale_in

        if not anisotropic:
            y = np.tanh(1.25 * f_in_norm) + 0.45 * f_in_norm
            y = y / (np.std(y) + 1e-8)
            x_clean = y
        else:
            raw_aniso = (1.15 * f_grad - 0.95 * f_curl + 0.65 * f_harm) / scale_in
            y = np.tanh(1.25 * raw_aniso) + 0.45 * raw_aniso
            y = y / (np.std(y) + 1e-8)
            x_clean = f_in_norm

        m = np.ones(n_e, dtype=np.float32)
        m[rng.choice(n_e, size=int(round(mask_ratio * n_e)), replace=False)] = 0.0
        x_noisy = (x_clean + rng.normal(0.0, noise_std, size=n_e)) * m
        xs.append(x_noisy.astype(np.float32))
        ys.append(y.astype(np.float32))
        ms.append(m)

    return {
        "x_obs": torch.tensor(np.array(xs))[..., None],
        "y_true": torch.tensor(np.array(ys))[..., None],
        "mask_obs": torch.tensor(np.array(ms))[..., None],
    }


def _r2_missing(pred: torch.Tensor, y_true: torch.Tensor, mask_obs: torch.Tensor, all_edges: bool) -> float:
    if all_edges:
        err = (pred - y_true).reshape(-1).detach().numpy()
        yt = y_true.reshape(-1).detach().numpy()
    else:
        miss = (1.0 - mask_obs) > 0.5
        err = (pred[miss] - y_true[miss]).detach().numpy()
        yt = y_true[miss].detach().numpy()
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _forward_hodge_operator(model: HodgeSimplicialNN, x: torch.Tensor, mask: torch.Tensor, residual_input: bool) -> torch.Tensor:
    if residual_input:
        return model(x, mask=mask)
    x_3d, m_3d, orig_mode = _format_edge_input(x, mask)
    feat = model.encode_edges(x_3d, mask=m_3d)
    out = model.out_proj(feat)
    return _restore_edge_output(out, orig_mode)


def _seed_worker(args: Tuple[bool, int, int]) -> Dict[str, Any]:
    torch.set_num_threads(1)
    anisotropic, seed, epochs = args
    sc_src = planar_punctured_complex(13, 9, hole_centers=((-0.44, 0.0), (0.44, 0.0)), hole_radius=0.25, seed=42)
    sc_tgt = planar_punctured_complex(16, 11, hole_centers=((-0.48, 0.08), (0.40, -0.08)), hole_radius=0.23, seed=1702)
    residual_input = not anisotropic
    all_edges = anisotropic

    tr = _generate_constitutive_dataset(sc_src, n_samples=150, seed=seed * 13 + 1, anisotropic=anisotropic)
    te_src = _generate_constitutive_dataset(sc_src, n_samples=64, seed=seed * 13 + 2, anisotropic=anisotropic)
    te_tgt = _generate_constitutive_dataset(sc_tgt, n_samples=64, seed=seed * 13 + 3, anisotropic=anisotropic)

    torch.manual_seed(seed)
    m_hodge = HodgeSimplicialNN(sc_src, hidden_channels=32, num_layers=4, filter_order=2)
    torch.manual_seed(seed)
    m_tied = TiedL1SimplicialNN(sc_src, hidden_channels=40, num_layers=4, filter_order=2)
    torch.manual_seed(seed)
    m_ulg = EdgeLineGraphGCN(sc_src, hidden_channels=48, num_layers=4)

    res: Dict[str, Any] = {"seed": seed, "anisotropic": anisotropic}
    for name, net in (("hodge_snn", m_hodge), ("tied_l1_snn", m_tied), ("unsigned_lg_gcn", m_ulg)):
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            if name == "hodge_snn":
                pred = _forward_hodge_operator(net, tr["x_obs"], tr["mask_obs"], residual_input=residual_input)
            elif name == "tied_l1_snn":
                pred = net(tr["x_obs"], mask=tr["mask_obs"], residual_input=residual_input)
            else:
                pred = net(tr["x_obs"], mask=tr["mask_obs"])
            if all_edges:
                loss = torch.mean((pred - tr["y_true"]) ** 2)
            else:
                w = 1.0 + 2.0 * (1.0 - tr["mask_obs"])
                loss = torch.mean(w * (pred - tr["y_true"]) ** 2)
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            if name == "hodge_snn":
                p_s = _forward_hodge_operator(net, te_src["x_obs"], te_src["mask_obs"], residual_input=residual_input)
                net_t = net.with_edge_orientations(sc_tgt, np.ones(sc_tgt.num_edges))
                net_t.eval()
                p_t = _forward_hodge_operator(net_t, te_tgt["x_obs"], te_tgt["mask_obs"], residual_input=residual_input)
            elif name == "tied_l1_snn":
                p_s = net(te_src["x_obs"], mask=te_src["mask_obs"], residual_input=residual_input)
                net_t = net.rebind_mesh(sc_tgt)
                net_t.eval()
                p_t = net_t(te_tgt["x_obs"], mask=te_tgt["mask_obs"], residual_input=residual_input)
            else:
                p_s = net(te_src["x_obs"], mask=te_src["mask_obs"])
                net_t = EdgeLineGraphGCN(sc_tgt, hidden_channels=48, num_layers=4)
                src_p = dict(net.named_parameters())
                for k, p in net_t.named_parameters():
                    p.data.copy_(src_p[k].data)
                net_t.eval()
                p_t = net_t(te_tgt["x_obs"], mask=te_tgt["mask_obs"])

            res[f"{name}_src"] = _r2_missing(p_s, te_src["y_true"], te_src["mask_obs"], all_edges=all_edges)
            res[f"{name}_tgt"] = _r2_missing(p_t, te_tgt["y_true"], te_tgt["mask_obs"], all_edges=all_edges)
    return res


def _paired_stats(a: List[float], b: List[float], label: str) -> Dict[str, Any]:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    diff = arr_a - arr_b
    md, lo, hi = paired_bootstrap_ci(diff, n_boot=10000, seed=2026)
    p_raw = 1.0 if np.allclose(diff, 0.0) else float(wilcoxon(diff, alternative="two-sided").pvalue)
    return {
        "contrast": label,
        "mean_a": float(arr_a.mean()),
        "sd_a": float(arr_a.std(ddof=1)),
        "mean_b": float(arr_b.mean()),
        "sd_b": float(arr_b.std(ddof=1)),
        "mean_diff": md,
        "ci95": [lo, hi],
        "p_raw": p_raw,
    }


def main() -> None:
    t0 = time.perf_counter()
    sc_src = planar_punctured_complex(13, 9, hole_centers=((-0.44, 0.0), (0.44, 0.0)), hole_radius=0.25, seed=42)
    seeds = [101 + 17 * k for k in range(10)]
    jobs = [(False, s, 140) for s in seeds] + [(True, s, 140) for s in seeds]

    print(f"[Simplicial Control] Running {len(jobs)} paired jobs sequentially...", flush=True)
    rows = []
    for idx_j, jb in enumerate(jobs, 1):
        r = _seed_worker(jb)
        rows.append(r)
        if idx_j % 5 == 0:
            print(f"  completed {idx_j}/{len(jobs)} ({time.perf_counter()-t0:.1f}s)", flush=True)

    def _pack(aniso_flag: bool) -> Dict[str, Any]:
        sub = sorted([r for r in rows if r["anisotropic"] == aniso_flag], key=lambda r: r["seed"])
        return {
            "params": {
                "HodgeSimplicialNN_h32": _param_count(HodgeSimplicialNN(sc_src, hidden_channels=32)),
                "TiedL1SimplicialNN_h40": _param_count(TiedL1SimplicialNN(sc_src, hidden_channels=40)),
                "UnsignedLG_EdgeGCN_h48": _param_count(EdgeLineGraphGCN(sc_src, hidden_channels=48)),
            },
            "source_244": {
                "hodge_snn": [r["hodge_snn_src"] for r in sub],
                "tied_l1_snn": [r["tied_l1_snn_src"] for r in sub],
                "unsigned_lg_gcn": [r["unsigned_lg_gcn_src"] for r in sub],
            },
            "unseen_shifted_398": {
                "hodge_snn": [r["hodge_snn_tgt"] for r in sub],
                "tied_l1_snn": [r["tied_l1_snn_tgt"] for r in sub],
                "unsigned_lg_gcn": [r["unsigned_lg_gcn_tgt"] for r in sub],
            },
        }

    iso = _pack(False)
    aniso = _pack(True)

    contrasts = [
        _paired_stats(iso["source_244"]["hodge_snn"], iso["source_244"]["tied_l1_snn"], "Isotropic Source (244): HodgeSNN - TiedL1SNN"),
        _paired_stats(iso["unseen_shifted_398"]["hodge_snn"], iso["unseen_shifted_398"]["tied_l1_snn"], "Isotropic Unseen (398): HodgeSNN - TiedL1SNN"),
        _paired_stats(iso["unseen_shifted_398"]["tied_l1_snn"], iso["unseen_shifted_398"]["unsigned_lg_gcn"], "Isotropic Unseen (398): TiedL1SNN - UnsignedLG"),
        _paired_stats(aniso["source_244"]["hodge_snn"], aniso["source_244"]["tied_l1_snn"], "Anisotropic Source (244): HodgeSNN - TiedL1SNN"),
        _paired_stats(aniso["unseen_shifted_398"]["hodge_snn"], aniso["unseen_shifted_398"]["tied_l1_snn"], "Anisotropic Unseen (398): HodgeSNN - TiedL1SNN"),
        _paired_stats(aniso["unseen_shifted_398"]["hodge_snn"], aniso["unseen_shifted_398"]["unsigned_lg_gcn"], "Anisotropic Unseen (398): HodgeSNN - UnsignedLG"),
    ]
    for c, pa in zip(contrasts, holm_adjust([x["p_raw"] for x in contrasts])):
        c["p_holm"] = pa

    # Compute exact paired inference on mesh_transfer.json (actual schema: mt["meshes"][m_name])
    mt_path = ROOT / "results" / "mesh_transfer.json"
    mt_inference: Dict[str, Any] = {}
    if mt_path.exists():
        mt = json.loads(mt_path.read_text(encoding="utf-8"))
        heldout_meshes = ("coarse", "shifted", "fine")
        harm_hits, hodge_prob_hits, s42_hits_list = [], [], []
        vortex_sse_hodge, vortex_sse_iso, vortex_sse_low = [], [], []
        for m_name in heldout_meshes:
            m_rec = mt["meshes"][m_name]
            y_true = np.asarray(m_rec["class_labels"], dtype=int)
            harm_pred = np.asarray(m_rec["loop_predictions_by_seed"]["42"]["harmonic_centroid"], dtype=int)
            harm_hits.extend((harm_pred == y_true).astype(float).tolist())
            seed_hits = np.mean(
                [(np.asarray(m_rec["loop_predictions_by_seed"][s]["hodge_snn"], dtype=int) == y_true).astype(float) for s in ("42", "101", "202")],
                axis=0,
            )
            hodge_prob_hits.extend(seed_hits.tolist())
            s42_hits_list.extend((np.asarray(m_rec["loop_predictions_by_seed"]["42"]["hodge_snn"], dtype=int) == y_true).tolist())

            miss_cnt = float(m_rec["localized_flux_linear"]["low_hodge"]["missing_count"]) / 60.0
            vortex_sse_hodge.extend((np.mean([m_rec["localized_flux_neural_by_seed"][s]["hodge_snn"]["sse_by_signal"] for s in ("42", "101", "202")], axis=0) / miss_cnt).tolist())
            vortex_sse_iso.extend((np.mean([m_rec["localized_flux_neural_by_seed"][s]["isotropic_l1"]["sse_by_signal"] for s in ("42", "101", "202")], axis=0) / miss_cnt).tolist())
            vortex_sse_low.extend((np.asarray(m_rec["localized_flux_linear"]["low_hodge"]["sse_by_signal"], dtype=float) / miss_cnt).tolist())

        harm_arr = np.asarray(harm_hits)
        hodge_arr = np.asarray(hodge_prob_hits)
        s42_hits = np.asarray(s42_hits_list, dtype=bool)
        loop_md, loop_lo, loop_hi = paired_bootstrap_ci(100.0 * (harm_arr - hodge_arr), n_boot=10000, seed=2026)
        n_harm_only = int(np.sum((harm_arr == 1.0) & (~s42_hits)))
        n_hodge_only = int(np.sum((harm_arr == 0.0) & s42_hits))
        p_mcnemar = float(binomtest(n_harm_only, n_harm_only + n_hodge_only, 0.5).pvalue) if (n_harm_only + n_hodge_only) > 0 else 1.0

        vh = np.asarray(vortex_sse_hodge)
        vi = np.asarray(vortex_sse_iso)
        vl = np.asarray(vortex_sse_low)
        md_vi, lo_vi, hi_vi = paired_bootstrap_ci(vi - vh, n_boot=10000, seed=2026)
        md_vl, lo_vl, hi_vl = paired_bootstrap_ci(vl - vh, n_boot=10000, seed=2026)
        p_vi = float(wilcoxon(vi - vh).pvalue)
        p_vl = float(wilcoxon(vl - vh).pvalue)
        adj_mt = holm_adjust([p_mcnemar, p_vi, p_vl])
        mt_inference = {
            "heldout_480_loops_harmonic_minus_hodge_pp": {
                "harmonic_acc_pct": float(100.0 * harm_arr.mean()),
                "hodge_mean_acc_pct": float(100.0 * hodge_arr.mean()),
                "mean_diff_pp": loop_md,
                "ci95_pp": [loop_lo, loop_hi],
                "seed42_discordant_harm_only": n_harm_only,
                "seed42_discordant_hodge_only": n_hodge_only,
                "p_raw": p_mcnemar,
                "p_holm": adj_mt[0],
            },
            "heldout_180_vortex_signals_mse_reduction_vs_conflated_lg": {
                "mean_mse_hodge": float(vh.mean()),
                "mean_mse_conflated": float(vi.mean()),
                "mean_mse_reduction": md_vi,
                "ci95_mse_reduction": [lo_vi, hi_vi],
                "p_raw": p_vi,
                "p_holm": adj_mt[1],
            },
            "heldout_180_vortex_signals_mse_reduction_vs_low_hodge": {
                "mean_mse_hodge": float(vh.mean()),
                "mean_mse_low_hodge": float(vl.mean()),
                "mean_mse_reduction": md_vl,
                "ci95_mse_reduction": [lo_vl, hi_vl],
                "p_raw": p_vl,
                "p_holm": adj_mt[2],
            },
        }

    out = {
        "seeds": seeds,
        "isotropic_imputation_control": iso,
        "anisotropic_constitutive_operator": aniso,
        "paired_10seed_contrasts": contrasts,
        "mesh_transfer_paired_inference": mt_inference,
    }
    out_path = ROOT / "results" / "anisotropic_constitutive_control.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Simplicial Control] Saved {out_path} in {time.perf_counter() - t0:.1f}s")
    for c in contrasts:
        print(
            f"  {c['contrast']}: {c['mean_a']:.4f} vs {c['mean_b']:.4f} | "
            f"diff={c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | Holm p={c['p_holm']:.4g}"
        )


if __name__ == "__main__":
    main()
