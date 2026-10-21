"""The self-previous condition is Z plus the replier's previous turn, without partner delivery."""

from pathlib import Path

import yaml

from src.experiments.layer1 import COMPARISONS


def test_self_previous_condition_is_implemented_as_registered_in_the_config():
    cfg = Path(__file__).resolve().parents[1] / "configs" / "j1_conditions.yaml"
    j1 = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    cond = {c["name"]: c["inputs"] for c in j1["conditions"]}
    assert "partner_delivery" not in cond["plus_self_previous"]
    assert COMPARISONS["self_previous_over_partner_text"] == ("plus_self_previous",
                                                              "plus_partner_text")
