"""The eight reported figures: six scientific, two technical. Exactly eight.

Every data figure is drawn **only from stored run outputs** (the JSON and NPZ
files a run directory holds), so it regenerates without manual editing and
never from numbers typed into a notebook. The two diagrams (Figure 1, Figure 7)
are drawn from the tracked configuration, not from results.

1. design diagram — what is in Z, what A is, what each control destroys;
2. detectability curve and null calibration (both layers' MDE on one panel;
   placed before any result);
3. observed statistic against the N1 and N2 null distributions;
4. per-block Δ with session-clustered intervals and the per-block MDE;
5. coupling map from the reduced-rank factors, with stability shading;
6. function × coupling interaction (Level E);
7. neural architecture diagram;
8. SSL layer-mixture weights α and the interpretable-probe profile.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import TARGET_BLOCKS, TARGET_DIMENSIONS
from src.viz import style

BLOCK_NAMES = list(TARGET_BLOCKS)


def _content(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["content"]


def _label(fig, data_kind: str) -> None:
    if data_kind != "sample":
        fig.text(0.99, 0.01, f"{data_kind} data — not a confirmatory result", ha="right",
                 va="bottom", fontsize=7, color=style.INK_SECONDARY)


def _box(ax, x, y, w, h, text, color):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec=style.INK,
                                lw=0.6, alpha=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, wrap=True)


def figure1_design():
    """What the model sees, what is predicted, and what each control destroys."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    _box(ax, 0.2, 3.3, 2.6, 1.2, "Z: reply text (PCA-64), reply words & syllables,\npartner "
         "text (PCA-64), replier, position, loudness offset", "#e8eef9")
    _box(ax, 0.2, 1.7, 2.6, 1.0, "A: partner delivery\n(8 dims, standardised per speaker)",
         "#fde7dc")
    _box(ax, 4.0, 2.4, 2.0, 1.2, "ridge / RRR\n(λ, rank by grouped\nnested CV)", "#eef7f1")
    _box(ax, 7.2, 2.4, 2.6, 1.2, "Y: reply delivery\n8 dims, 4 blocks\n(standardised, "
         "orthogonalised)", "#fff4d6")
    ax.annotate("", (4.0, 3.0), (2.8, 3.9), arrowprops={"arrowstyle": "->"})
    ax.annotate("", (4.0, 3.0), (2.8, 2.2), arrowprops={"arrowstyle": "->"})
    ax.annotate("", (7.2, 3.0), (6.0, 3.0), arrowprops={"arrowstyle": "->"})
    _box(ax, 0.2, 0.2, 4.6, 1.0, "N1: A from another dyad, same split, matched on duration "
         "decile and word count — destroys the pairing", "#f2f2f2")
    _box(ax, 5.1, 0.2, 4.7, 1.0, "N2: A from the same speaker, same interaction, ≥ 10 turns "
         "away — destroys local adjacency only", "#f2f2f2")
    ax.set_title("Figure 1. Design: Δ = held-out gain from adding A to Z, against donor nulls")
    return fig


def figure2_detectability(j3_dir: Path, neural_detectability: Path | None = None):
    import matplotlib.pyplot as plt

    det = _content(Path(j3_dir) / "detectability.json")
    g1 = _content(Path(j3_dir) / "gate_G1.json")
    status = json.loads((Path(j3_dir) / "status.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    for k, (name, c) in enumerate(det["curves"].items()):
        ax.plot(c["targets"], c["power_aggregate"], marker="o", color=style.SERIES[k],
                label=f"Layer 1, {name.replace('_', ' ')}")
    if det["mde"]["aggregate"] is not None:
        ax.axvline(det["mde"]["aggregate"], color=style.SERIES[0], ls=":", lw=1)
    if neural_detectability is not None:
        nd = _content(Path(neural_detectability))
        ax.plot(nd["targets"], nd["power_aggregate"], marker="s", color=style.SERIES[2],
                label="Layer 2 (neural)")
        if nd.get("mde") is not None:
            ax.axvline(nd["mde"], color=style.SERIES[2], ls=":", lw=1)
    ax.axhline(det["mde"]["power_level"], color=style.INK_SECONDARY, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("injected block partial R²")
    ax.set_ylabel("rejection rate vs N1")
    ax.set_title("Detectability (MDE marked)")
    ax.legend()
    ax = axes[1]
    rates = g1["evidence"]["rates"]
    for k, r in enumerate(rates.values()):
        err = [[r["rate"] - r["ci95"][0]], [r["ci95"][1] - r["rate"]]]
        ax.errorbar([k], [r["rate"]], yerr=err, fmt="o", color=style.SERIES[k], capsize=4)
    ax.axhline(g1["evidence"]["alpha"], color=style.INK_SECONDARY, lw=0.8)
    ax.set_xticks(range(len(rates)), list(rates))
    ax.set_ylabel("rejection rate at β = 0")
    ax.set_title(f"Null calibration (G1: {g1['status']})")
    _label(fig, status["data_kind"])
    return fig


def figure3_null_distributions(j2_dir: Path):
    import matplotlib.pyplot as plt

    z = np.load(Path(j2_dir) / "null_draws.npz")
    tests = _content(Path(j2_dir) / "j2_tests.json")
    status = json.loads((Path(j2_dir) / "status.json").read_text())
    obs = tests["observed"]["aggregate_partial_r2"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    key = {"N1": "H1_vs_N1", "N2": "H4_vs_N2"}
    for k, s in enumerate(("N1", "N2")):
        ax.hist(z[f"{s}_aggregate"], bins=40, alpha=0.6, color=style.SERIES[k],
                label=f"{s} null (p = {tests[key[s]]['reported']})")
    ax.axvline(obs, color=style.INK, lw=1.5, label="observed")
    ax.set_xlabel("aggregate partial R² (equal weight per block)")
    ax.set_ylabel("donor draws")
    ax.set_title("Figure 3. Observed statistic against the N1 and N2 nulls")
    ax.legend()
    _label(fig, status["data_kind"])
    return fig


def figure4_block_delta(j1_dir: Path, j3_dir: Path):
    import matplotlib.pyplot as plt

    s = _content(Path(j1_dir) / "j1_summary.json")
    det = _content(Path(j3_dir) / "detectability.json")
    status = json.loads((Path(j1_dir) / "status.json").read_text())
    comp = s["comparisons"]["partner_delivery_over_partner_text"]
    pr2 = np.array(comp["block_partial_r2"])
    ci = np.array(comp["intervals"]["0.95"]["block"] if "0.95" in comp["intervals"]
                  else comp["intervals"][0.95]["block"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(pr2))
    ax.errorbar(x, pr2, yerr=[pr2 - ci[:, 0], ci[:, 1] - pr2], fmt="o", color=style.SERIES[0],
                capsize=4, label="Δ (95% session-bootstrap interval)")
    mde = det["mde"]["block"]
    for i, m in enumerate(mde):
        if m is not None:
            ax.plot([i - 0.3, i + 0.3], [m, m], color=style.SERIES[1], lw=2,
                    label="block MDE" if i == 0 else None)
    ax.axhline(0, color=style.INK_SECONDARY, lw=0.8)
    ax.set_xticks(x, [b.replace("_", " ") for b in BLOCK_NAMES])
    ax.set_ylabel("block partial R²")
    ax.set_title("Figure 4. Per-block Δ with the detectability floor")
    ax.legend()
    _label(fig, status["data_kind"])
    return fig


def figure5_coupling_map(j1_dir: Path):
    import matplotlib.pyplot as plt

    cm = _content(Path(j1_dir) / "coupling_map.json")
    status = json.loads((Path(j1_dir) / "status.json").read_text())
    P = np.array(cm["partner_loadings"])
    Q = np.array(cm["reply_loadings"])
    coupling = P @ Q.T                                   # partner dim × reply dim
    fig, ax = plt.subplots(figsize=(6.4, 5))
    lim = np.abs(coupling).max() or 1.0
    im = ax.imshow(coupling, cmap="RdBu_r", vmin=-lim, vmax=lim)
    names = [d.replace("_", " ") for d in TARGET_DIMENSIONS]
    ax.set_xticks(range(8), names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(8), names, fontsize=7)
    ax.set_xlabel("reply dimension")
    ax.set_ylabel("partner dimension")
    # Stability is shown as subspace similarities, not per-loading signs,
    # which flip at random for loadings near zero.
    st = cm.get("stability", {})
    note = (f"bootstrap similarity reply {st['bootstrap_reply']:.2f}, partner "
            f"{st['bootstrap_partner']:.2f}" if st else "stability not computed")
    ax.set_title(f"Figure 5. Coupling map, rank {cm['rank']} ({note}; descriptive)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7)
    _label(fig, status["data_kind"])
    return fig


def figure6_function_interaction(j4_dir: Path):
    import matplotlib.pyplot as plt

    ps = _content(Path(j4_dir) / "per_stratum_delta.json")["strata"]
    status = json.loads((Path(j4_dir) / "status.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for k, v in enumerate(ps.values()):
        iv = v["intervals"]["0.95"]["aggregate"] if "0.95" in v["intervals"] else \
            v["intervals"][0.95]["aggregate"]
        ax.errorbar([k], [v["aggregate_partial_r2"]], yerr=[[v["aggregate_partial_r2"] - iv[0]],
                                                            [iv[1] - v["aggregate_partial_r2"]]],
                    fmt="o", color=style.SERIES[k % len(style.SERIES)], capsize=4)
    ax.set_xticks(range(len(ps)), [c.replace("_", " ") for c in ps], rotation=15, fontsize=7)
    ax.axhline(0, color=style.INK_SECONDARY, lw=0.8)
    ax.set_ylabel("aggregate partial R² within stratum")
    ax.set_title("Figure 6. Δ by conversational function (descriptive; the test is the "
                 "interaction term)", fontsize=9)
    _label(fig, status["data_kind"])
    return fig


def figure7_architecture():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    _box(ax, 0.1, 4.6, 2.4, 1.0, "frozen WavLM L3/6/9/12\n+ HuBERT (PCA-128)", "#f2f2f2")
    _box(ax, 2.9, 4.6, 2.2, 1.0, "SSL layer mixer\nα = softmax(w), → 192", "#e8eef9")
    _box(ax, 5.5, 4.6, 2.6, 1.0, "temporal transformer\n3 pre-norm blocks, 4 heads\nmask, "
         "attention pool → a_i", "#e8eef9")
    _box(ax, 0.1, 2.8, 2.4, 0.9, "partner text (64) → MLP → p_i", "#fde7dc")
    _box(ax, 0.1, 1.6, 2.4, 0.9, "reply text + metadata\n+ speaker → MLP → c_i", "#fde7dc")
    _box(ax, 3.4, 2.0, 2.8, 1.4, "cross-attention ([c;p] → frames)\n+ a_i residual\nconcat "
         "[c; p; attended]", "#eef7f1")
    _box(ax, 7.0, 2.0, 2.6, 1.4, "2-layer MLP head\n8 means (+8 log-variances)", "#fff4d6")
    for a, b in (((2.5, 5.1), (2.9, 5.1)), ((5.1, 5.1), (5.5, 5.1)), ((6.8, 4.6), (5.5, 3.4)),
                 ((2.5, 3.2), (3.4, 2.9)), ((2.5, 2.0), (3.4, 2.4)), ((6.2, 2.7), (7.0, 2.7))):
        ax.annotate("", b, a, arrowprops={"arrowstyle": "->"})
    ax.text(5, 0.6, "Same parameters in every condition: the partner pathway receives SSL "
            "frames, donor frames, a learned null-token sequence, or the 8-D vector.",
            ha="center", fontsize=8)
    ax.set_title("Figure 7. Layer-2 architecture (one block of cross-attention; generates "
                 "nothing)")
    return fig


def figure8_alpha_and_probe(neural_dir: Path):
    import matplotlib.pyplot as plt

    hist = _content(Path(neural_dir) / "history.json")
    probe = _content(Path(neural_dir) / "probes.json")
    status = json.loads((Path(neural_dir) / "status.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    from src.models.ssl_layer_mixer import SOURCES

    axes[0].bar(range(len(SOURCES)), hist["alpha"], color=style.SERIES[0])
    axes[0].set_xticks(range(len(SOURCES)), [s.replace("_", " ") for s in SOURCES], fontsize=7)
    axes[0].set_ylabel("α")
    axes[0].set_title("Learned mixture weights (a modelling statement, not a finding)",
                      fontsize=9)
    r2 = probe["per_dimension_r2"]
    axes[1].bar(range(len(r2)), r2, color=style.SERIES[1])
    axes[1].set_xticks(range(len(r2)), [d.replace("_", " ") for d in TARGET_DIMENSIONS],
                       rotation=60, ha="right", fontsize=7)
    axes[1].set_ylabel("probe R² (held-out speakers)")
    axes[1].set_title("Interpretable probe profile", fontsize=9)
    _label(fig, status["data_kind"])
    return fig


FIGURES = (figure1_design, figure2_detectability, figure3_null_distributions,
           figure4_block_delta, figure5_coupling_map, figure6_function_interaction,
           figure7_architecture, figure8_alpha_and_probe)

__all__ = [f.__name__ for f in FIGURES] + ["FIGURES"]
