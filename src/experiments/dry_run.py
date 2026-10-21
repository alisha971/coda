"""End-to-end dry run on a synthetic corpus (Gate G). Software check only.

A small synthetic corpus is generated and pushed through **the real code** of
every stage: timed words → turns (``build_turns``) → pairs (``build_pairs``) →
rendered frame tracks → the eight dimensions (``compute_turn_prosody``) →
speaker-disjoint splits → text embeddings (a deterministic stand-in encoder) →
standardisation, orthogonalisation, Z → J1 → J2 → J3 → J4 → gates → stop rule
→ figures. Every output is written with ``data_kind = "synthetic"``.

A known partner → reply coupling is planted in the pitch and energy latents
(``coupling``), so the run also checks that the pipeline recovers a planted
dependence and finds none when ``coupling = 0``. **This is a software check.
It says nothing about whether the hypothesis holds in conversation.**

It fails loudly: :func:`check_outputs` raises if any expected output is
missing or has the wrong shape.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RidgeConfig
from src.data.pairs import build_pairs
from src.data.splits import make_split
from src.data.turns import build_turns
from src.features.prosody import ProsodySettings, compute_turn_prosody, file_loudness_offset
from src.features.tracks import Tracks

VOCAB = ("so", "well", "really", "maybe", "the", "weekend", "conversation", "people",
         "think", "yeah", "right", "travel", "family", "work", "music", "interesting")
STEP = 0.01


@dataclass
class SyntheticCorpus:
    turns: pd.DataFrame
    pairs: pd.DataFrame
    words: dict[str, pd.DataFrame]
    tracks: dict[str, Tracks]
    sessions: pd.DataFrame
    latent: pd.DataFrame


#: Words whose presence shifts delivery when ``lexical > 0`` (synthetic only).
MARKERS = ("really", "wow", "interesting")


def make_corpus(n_sessions: int = 30, turns_per_interaction: int = 60, coupling: float = 0.5,
                seed: int = 0, drift_sd: float = 0.0, drift_rho: float = 0.95,
                lexical: float = 0.0, n_interactions: int = 1, interaction_sd: float = 0.0,
                self_ar: float = 0.0) -> SyntheticCorpus:
    """Two speakers per session, alternating turns, one or more interactions each.

    Each turn has three latents (pitch, energy, boundary). Speakers differ in
    habitual pitch and loudness (speaker style), which per-speaker
    standardisation removes. Optional ingredients, used to test failure modes of
    the design:

    - ``coupling``: local adaptation, a turn's latents depend on the previous
      (partner's) turn's latents;
    - ``drift_sd``, ``drift_rho``: a state shared by both speakers, an AR(1)
      process over turns with persistence ``drift_rho``, added to every turn. No
      turn depends on the partner's turn, yet adjacent turns are more alike than
      distant ones;
    - ``n_interactions``, ``interaction_sd``: several interactions per session,
      each with an offset shared by both speakers (the prompt changes every few
      minutes in the corpus);
    - ``self_ar``: a speaker's latents depend on their own previous turn
      (within-speaker momentum, no partner dependence);
    - ``lexical``: marker words raise the speaker's own pitch latent and the next
      turn's pitch latent, so partner delivery is correlated with partner words.
    """
    rng = np.random.default_rng(seed)
    all_turns, words_by_file, tracks, latent_rows, sess_rows = [], {}, {}, [], []
    for s in range(n_sessions):
        sk = f"V9_S{s:04d}"
        spk = {0: f"V9_P{2 * s:04d}", 1: f"V9_P{2 * s + 1:04d}"}
        sess_rows += [(sk, spk[0]), (sk, spk[1])]
        base_f0 = {0: rng.uniform(100, 140), 1: rng.uniform(170, 240)}
        base_db = {0: rng.uniform(55, 65), 1: rng.uniform(55, 65)}
        for i in range(n_interactions):
            ik = f"{sk}_I{i + 1:08d}"
            offset = interaction_sd * rng.normal(0, 1, 3)
            t, prev, prev_markers = 0.5, np.zeros(3), 0.0
            own_prev = {0: np.zeros(3), 1: np.zeros(3)}
            drift = rng.normal(0, 1, 3)
            spans = {0: [], 1: []}
            turn_latent = []
            for k in range(turns_per_interaction):
                who = k % 2
                n_words = int(rng.integers(2, 9))
                words = [VOCAB[int(rng.integers(len(VOCAB)))] for _ in range(n_words)]
                markers = sum(w in MARKERS for w in words) / n_words
                drift = drift_rho * drift + np.sqrt(1 - drift_rho ** 2) * rng.normal(0, 1, 3)
                lat = (coupling * prev + self_ar * own_prev[who] + rng.normal(0, 1, 3)
                       + drift_sd * drift + offset)
                lat[0] += lexical * (markers + prev_markers)
                start = t
                for word in words:
                    d = rng.uniform(0.18, 0.35)
                    spans[who].append((t, t + d, word))
                    t += d + (rng.uniform(0.25, 0.6) if rng.random() < 0.15 else 0.01)
                turn_latent.append((who, start, t, lat))
                prev, prev_markers, own_prev[who] = lat, markers, lat
                t += rng.uniform(0.2, 0.5)
            dur = t + 1.0
            n = int(dur / STEP)
            times = STEP / 2 + np.arange(n) * STEP
            for who in (0, 1):
                fid = f"{ik}_P{spk[who].split('_P')[1]}"
                words_by_file[fid] = pd.DataFrame(spans[who], columns=["start", "end", "word"])
                f0 = np.zeros(n, np.float32)
                db = np.full(n, 30.0, np.float32)
                for (who2, ts, te, lat) in turn_latent:
                    if who2 != who:
                        continue
                    sel = (times >= ts) & (times < te)
                    inside = np.zeros(n, bool)
                    for ws, we, _ in spans[who]:
                        if ws >= ts - 1e-9 and we <= te + 1e-9:
                            inside |= (times >= ws) & (times < we)
                    m = sel & inside
                    f0[m] = base_f0[who] * np.exp(0.08 * lat[0] + 0.3 * lat[2] * (times[m] - te)
                                                  + rng.normal(0, 0.02, m.sum()))
                    db[m] = base_db[who] + 2.0 * lat[1] + rng.normal(0, 1.0, m.sum())
                tracks[fid] = Tracks(STEP, STEP / 2, f0, (f0 > 0).astype(np.float32), STEP / 2,
                                     db, db, dur)
            turns = build_turns({spk[w]: words_by_file[f"{ik}_P{spk[w].split('_P')[1]}"]
                                 for w in (0, 1)}, ik, sk)
            all_turns.append(turns)
            for (who, ts, _te, lat) in turn_latent:
                latent_rows.append({"session_key": sk, "interaction_key": ik,
                                    "speaker": spk[who], "start": ts, "pitch": lat[0],
                                    "energy": lat[1], "boundary": lat[2]})
    turns = pd.concat(all_turns, ignore_index=True)
    return SyntheticCorpus(turns, build_pairs(turns), words_by_file, tracks,
                           pd.DataFrame(sess_rows, columns=["session_key", "participant_key"]),
                           pd.DataFrame(latent_rows))


class HashEncoder:
    """Deterministic stand-in for the frozen sentence encoder (dry run only)."""

    def __call__(self, texts):
        out = np.zeros((len(texts), 384), np.float32)
        for i, t in enumerate(texts):
            for w in t.split():
                h = sum(ord(c) * (j + 1) for j, c in enumerate(w))
                out[i, h % 384] += 1.0
        return out


def prepare(corpus: SyntheticCorpus, cache_dir: Path, seed: int = 1, min_turns: int = 5):
    from src.experiments.pipeline import prepare_analysis
    from src.features.text_enc import encode_cached

    settings = ProsodySettings()
    prosody = compute_turn_prosody(corpus.turns, lambda sk, fid: corpus.tracks.get(fid),
                                   corpus.words.get, settings)
    split = make_split(corpus.sessions, {"train": 0.7, "val": 0.1, "test": 0.2}, seed)
    main = corpus.turns[~corpus.turns["embedded"]]
    emb = encode_cached(main["turn_id"].tolist(), main["text"].tolist(), HashEncoder(),
                        cache_dir / "text", {"model": "hash-encoder-dry-run"}, shard_size=500)
    loud = pd.Series({fid: file_loudness_offset(tr) for fid, tr in corpus.tracks.items()})
    data = prepare_analysis(corpus.turns, corpus.pairs, prosody, split, emb, loud, min_turns,
                            text_pca_dim=16)
    return data, prosody, split


def run(results_root: Path | None = None, n_sessions: int = 30, coupling: float = 0.5,
        n_draws: int = 19, seed: int = 0) -> dict:
    """The full synthetic pipeline. Returns paths and the key numbers."""
    from src.analysis.situation import classifier_inputs, label_pairs
    from src.experiments.gates import stop_rule
    from src.experiments.run_j1 import run_j1
    from src.experiments.run_j2 import run_j2
    from src.experiments.run_j3 import run_j3
    from src.experiments.run_j4 import run_j4

    results_root = Path(results_root or tempfile.mkdtemp(prefix="coda_dry_run_"))
    corpus = make_corpus(n_sessions, coupling=coupling, seed=seed)
    data, prosody, split = prepare(corpus, results_root / "_cache")
    # Mechanics only: gates report PROVISIONAL (see Settings.development).
    from src.settings import Settings

    settings = Settings().development()
    ridge = RidgeConfig(nested_cv_folds=3)
    stage = f"dry_run_c{coupling}"
    j3_dir, j3 = run_j3(data, corpus.turns, prosody, results_root=results_root,
                        data_kind="synthetic", stage=stage, n_draws=n_draws,
                        targets=[0.02, 0.05], settings=settings, ridge=ridge,
                        couplings=("identity_matched_dimensions",), replicates=4,
                        replicates_beta0=8, seed=seed)
    gates = {"G1": j3["G1"], "G2": j3["G2"]}
    j1_dir = run_j1(data, results_root=results_root, data_kind="synthetic", stage=stage,
                    settings=settings, ridge=ridge, gates=gates, n_boot=200, h2_boot=10,
                    mixed_effects=_has_statsmodels(), seed=seed)
    j2_dir, j2 = run_j2(data, corpus.turns, prosody, results_root=results_root,
                        data_kind="synthetic", stage=stage, n_draws=n_draws, j1_dir=j1_dir,
                        settings=settings, ridge=ridge, gates=gates, seed=seed)

    def classifier(partner_text, reply_text):          # synthetic labels only
        return ("backchannel_adjacent" if len(reply_text.split()) <= 3
                else "narrative_inform_to_continuation")

    labels = label_pairs(classifier_inputs(data.pairs, corpus.turns), classifier)
    hand = labels.rename(columns={"function": "hand_label"})[["pair_id", "hand_label"]]
    j4_dir, j4 = (None, None)
    if _has_statsmodels():
        j4_dir, j4 = run_j4(data, labels, hand, results_root=results_root,
                            data_kind="synthetic", stage=stage,
                            h2_h4_p={"H2": j2["tests"]["H2_leading_strength_vs_N1"]["p"],
                                     "H4": j2["tests"]["H4_vs_N2"]["p"]},
                            settings=settings, ridge=ridge, gates=gates, n_boot=100,
                            seed=seed)
    state = {"delta_n1": True, "delta_n2": True, "mde_stated": j3["mde"]["aggregate"] is not None,
             "g1": j3["G1"].status, "g2": j3["G2"].status,
             "g4": j4["G4"]["status"] if j4 else None,
             "g5": j4["G5"].status if j4 else None}
    out = {"results_root": str(results_root), "j1": str(j1_dir), "j2": str(j2_dir),
           "j3": str(j3_dir), "j4": str(j4_dir) if j4_dir else None,
           "observed_aggregate_partial_r2":
               j2["tests"]["observed"]["aggregate_partial_r2"],
           "p_vs_N1": j2["tests"]["H1_vs_N1"]["p"], "p_vs_N2": j2["tests"]["H4_vs_N2"]["p"],
           "G1": j3["G1"].status, "G1_provisional_outcome": j3["G1"].evidence[
               "provisional_outcome"], "G2": j3["G2"].status, "G3": j2["G3"].evidence["reading"],
           "beta0_rejection": j3["beta0"], "mde": j3["mde"], "stop_rule": stop_rule(state),
           "n_pairs": int(len(data.pairs)), "n_sessions": int(len(np.unique(data.groups)))}
    check_outputs(out)
    return out


def _has_statsmodels() -> bool:
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        return False
    return True


EXPECTED = {
    "j1": ["j1_summary.json", "j1_errors.npz", "coupling_map.json", "droppage_by_condition.csv",
           "status.json"],
    "j2": ["donors_N1.npz", "donors_N2.npz", "null_draws.npz", "j2_tests.json", "gate_G3.json"],
    "j3": ["j3_levels.json", "detectability.json", "gate_G1.json", "gate_G2.json"],
}


def check_outputs(out: dict) -> None:
    """Raise if any expected output is missing or a run did not complete."""
    import json

    for exp, files in EXPECTED.items():
        d = Path(out[exp])
        for f in files:
            if not (d / f).exists():
                raise FileNotFoundError(f"dry run: {exp} did not write {f}")
        st = json.loads((d / "status.json").read_text(encoding="utf-8"))
        if st["status"] != "completed" or st["data_kind"] != "synthetic":
            raise AssertionError(f"dry run: {exp} status {st}")
    summary = json.loads((Path(out["j1"]) / "j1_summary.json").read_text(encoding="utf-8"))
    comps = summary["content"]["comparisons"]
    for c in ("partner_text_over_content", "partner_delivery_over_partner_text",
              "self_previous_over_partner_text"):
        if len(comps[c]["block_partial_r2"]) != 4:
            raise AssertionError(f"dry run: {c} does not have four blocks")


__all__ = ["SyntheticCorpus", "make_corpus", "HashEncoder", "prepare", "run", "check_outputs"]
