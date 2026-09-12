# CODA session log

What was executed, when, and with what outcome. Engineering record.

Distinct from `docs/research_journal.md`, which records what was expected, observed and learned. A run belongs here; what it meant belongs there.

---

## 2026-09-12 — Phase 0, repository foundation

**Environment.** Local machine: Intel i5-7200U, 2 cores / 4 threads, 7.9 GB RAM, 34.6 GB free on `C:`, 365.8 GB free on `G:`. GeForce 920MX, 2 GB, not used for any stage.

**Done.**

- Ignore rules corrected: artifact patterns anchored to the repository root so `src/data/`, `src/features/` and `src/models/` are tracked. Verified with `git check-ignore` in both directions across 40 paths.
- Experiment output directories moved from ignored to tracked, with binary extensions retained as the size guard.
- `src/config.py`: configuration objects, run-identifier hashing, runtime detection, capacity-control assertion.
- `tests/test_config.py`: 20 tests. All pass.
- `pyproject.toml`, `environment.yml`, `requirements.txt` written with exact pins.
- `docs/research_overview.md`, `PREREGISTRATION.md`, `CLAUDE.md`, `README.md` written.

**Test run.** `python -m pytest tests/ -q` — 20 passed.

**Notable.** The dimension-count test failed on first run, revealing a discrepancy in the governing specification rather than a coding error. See the research journal entry of the same date.

- `configs/`: eight condition and pipeline configuration files.
- `tests/test_configs_match_code.py`: 30 tests asserting the tracked YAML and the code cannot drift apart.
- `notebooks/00_setup_and_config.ipynb`: written, executed end to end, outputs stored.
- `CODA_GIT_COMMANDS.md` written. `CODA_v3_with_neural_layer.md` added to the ignore rules as a superseded internal document.

**Test run.** `python -m pytest tests/ -q` - 50 passed.

**Open item: interpreter version drift.** `environment.yml` pins Python 3.11, chosen for cloud-runtime compatibility, but the notebook was executed under the system interpreter at 3.13.15. The pinned environment has not been created locally yet. This is acceptable for setup work and **not** acceptable once any target value is computed: the cross-environment reproducibility argument depends on both environments resolving to the same library versions. Create the pinned environment before the pilot extraction.

**Not done.** The pinned environment has not been instantiated. No corpus access yet. No git remote yet, which the first cloud session requires.
