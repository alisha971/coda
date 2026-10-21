"""Pre-execution preflight: the human-controlled gate before any expensive run.

Before real-data prosody targets, J1-J4, SSL extraction or neural training run,
:func:`preflight` assembles a report and :func:`require_preflight` refuses to
continue unless every critical check passes. Synthetic dry runs skip the
real-data checks but still run the rest.

Checks:

1. **Integrity suite** — a recorded pytest run for *this* code fingerprint
   with no failures (the suite passes before any fitting job runs).
2. **Split manifest valid** — speaker-disjoint, every session assigned.
3. **Output is new** — the planned output directory does not exist.
4. **Gate prerequisites** — G1 has not failed; the label the step's outputs
   will carry (confirmatory or not) is stated.
5. **Compute estimate and smoke test** — recorded in the report.

Nothing here launches a job. A run starts only when its driver is called with
a passing report, which is a deliberate human action.
"""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class PreflightFailed(RuntimeError):
    """A real-data step was requested without a passing preflight report."""


@dataclass
class PreflightReport:
    step: str
    data_kind: str
    checks: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: str, critical: bool = True) -> None:
        self.checks.append({"check": name, "ok": bool(ok), "critical": critical,
                            "detail": detail})

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks if c["critical"])

    def as_dict(self) -> dict:
        return {"step": self.step, "data_kind": self.data_kind, "ok": self.ok,
                "checks": self.checks, "info": self.info,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def markdown(self) -> str:
        lines = [f"# Preflight — `{self.step}` on `{self.data_kind}` data", "",
                 f"**Result: {'PASS' if self.ok else 'BLOCKED'}**", "",
                 "| Check | Result | Critical | Detail |", "|---|---|---|---|"]
        for c in self.checks:
            lines.append(f"| {c['check']} | {'ok' if c['ok'] else 'FAIL'} | "
                         f"{'yes' if c['critical'] else 'no'} | {c['detail']} |")
        for k, v in self.info.items():
            lines += ["", f"**{k}.** `{json.dumps(v, default=str)}`"]
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Integrity suite record
# --------------------------------------------------------------------------


def record_integrity_suite(root: Path, python: str | None = None) -> dict:
    """Run the test suite and store its outcome keyed by the code fingerprint."""
    from src.runtime import code_fingerprint

    fp = code_fingerprint(root)
    out_dir = root / "results" / "integrity"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    xml = out_dir / f"pytest__{fp}__{stamp}.xml"
    proc = subprocess.run([python or sys.executable, "-m", "pytest", "tests/", "-q",
                           "-p", "no:cacheprovider", f"--junitxml={xml}"],
                          cwd=root, capture_output=True, text=True, check=False)
    suite = ET.parse(xml).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    ver = subprocess.run([python or sys.executable, "-c",
                          "import platform; print(platform.python_version())"],
                         capture_output=True, text=True, check=False).stdout.strip()
    # The interpreter's version is recorded, not its filesystem path.
    rec = {"code_fingerprint": fp, "created_utc": stamp, "python_version": ver,
           "tests": int(suite.get("tests")), "failures": int(suite.get("failures")),
           "errors": int(suite.get("errors")), "skipped": int(suite.get("skipped")),
           "returncode": proc.returncode, "tail": proc.stdout.strip().splitlines()[-3:]}
    (out_dir / f"pytest__{fp}__{stamp}.json").write_text(json.dumps(rec, indent=2),
                                                        encoding="utf-8")
    xml.unlink(missing_ok=True)
    return rec


def latest_integrity(root: Path, fingerprint: str) -> dict | None:
    recs = sorted((root / "results" / "integrity").glob(f"pytest__{fingerprint}__*.json"))
    return json.loads(recs[-1].read_text(encoding="utf-8")) if recs else None


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def preflight(step: str, data_kind: str, root: Path,
              split_manifest: pd.DataFrame | None = None,
              sessions: pd.DataFrame | None = None, planned_output: Path | None = None,
              gates: dict | None = None, compute_estimate: dict | None = None,
              smoke_test: str | None = None) -> PreflightReport:
    from src.experiments.gates import GateBlocked, require
    from src.runtime import code_fingerprint

    rep = PreflightReport(step, data_kind)
    real = data_kind != "synthetic"
    fp = code_fingerprint(root)
    rep.info["code_fingerprint"] = fp

    rec = latest_integrity(root, fp)
    rep.add("integrity suite passed for this code",
            rec is not None and rec["failures"] == 0 and rec["errors"] == 0,
            "no record for this fingerprint; run record_integrity_suite()" if rec is None
            else f"{rec['tests']} tests, {rec['failures']} failures, {rec['errors']} errors, "
                 f"{rec['skipped']} skipped ({rec['created_utc']})")

    if split_manifest is not None and sessions is not None:
        from src.data.splits import SplitLeakage, assert_disjoint

        try:
            assert_disjoint(split_manifest, sessions)
            rep.add("split manifest valid", True, "speaker-disjoint; every session assigned")
        except SplitLeakage as err:
            rep.add("split manifest valid", False, str(err))
    elif step not in ("prosody_targets",):
        rep.add("split manifest valid", False, "no split manifest supplied")

    if planned_output is not None:
        rep.add("output directory is new", not Path(planned_output).exists(),
                str(planned_output))

    if gates is not None:
        gate_step = {"j1_j2": "j1", "j4": "j4", "neural": "neural"}.get(step)
        if gate_step:
            try:
                label = require(gates, gate_step)
                rep.add("gate prerequisites", True, f"outputs will be labelled: {label}")
                rep.info["output_label"] = label
            except GateBlocked as err:
                rep.add("gate prerequisites", False, str(err))

    rep.add("compute estimate recorded", compute_estimate is not None,
            json.dumps(compute_estimate) if compute_estimate else "none supplied",
            critical=real)
    rep.add("smoke test defined", smoke_test is not None, smoke_test or "none supplied",
            critical=real)
    return rep


def require_preflight(report: PreflightReport | None, data_kind: str) -> None:
    """Called at the top of every expensive driver."""
    if data_kind == "synthetic":
        return
    if report is None:
        raise PreflightFailed("real-data runs need a preflight report (src.experiments.preflight)")
    if report.data_kind != data_kind:
        raise PreflightFailed("the preflight report was made for different data")
    if not report.ok:
        failed = [c["check"] for c in report.checks if c["critical"] and not c["ok"]]
        raise PreflightFailed(f"preflight blocked: {failed}")


__all__ = ["PreflightFailed", "PreflightReport", "record_integrity_suite", "latest_integrity",
           "preflight", "require_preflight"]
