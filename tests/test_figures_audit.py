"""The eight figures regenerate from stored outputs; the leakage register cites real tests."""

from pathlib import Path

import pytest

from src.analysis.audit import LEAKAGE_REGISTER, verify_register

ROOT = Path(__file__).resolve().parents[1]


def test_exactly_eight_figure_functions():
    """The figure library is eight functions."""
    from src.viz.figures import FIGURES

    assert len(FIGURES) == 8


def test_every_leakage_register_row_has_evidence_that_exists():
    df = verify_register(ROOT)
    assert set(df["route"]) == {r for r, _, _ in LEAKAGE_REGISTER}
    assert len({r for r, _, _ in LEAKAGE_REGISTER}) == 17


@pytest.mark.integration
def test_data_figures_render_from_a_dry_run(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("statsmodels")
    import matplotlib

    matplotlib.use("Agg")
    from src.experiments.dry_run import run
    from src.viz import figures as F

    out = run(tmp_path, n_sessions=20, coupling=0.5, n_draws=9)
    for fig in (F.figure1_design(), F.figure7_architecture(),
                F.figure2_detectability(Path(out["j3"])),
                F.figure3_null_distributions(Path(out["j2"])),
                F.figure4_block_delta(Path(out["j1"]), Path(out["j3"])),
                F.figure5_coupling_map(Path(out["j1"])),
                F.figure6_function_interaction(Path(out["j4"]))):
        assert fig.axes
