"""Guard against PLAN.md drifting away from the artifacts it quotes.

Figures in the roadmap are re-derived on every pipeline run, and it is easy for
a hand-written number to survive a model change. This asserts that the headline
metrics PLAN.md prints still match reports/, at the precision Stage 7g says
each one supports.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
    su, ce, un = (load("suitability_metrics.json"), load("certification.json"),
                  load("uncertainty.json"))
    if not all((su, ce, un)):
        print("reports missing; skipping (run the pipeline first)")
        return 0

    dp = {k: v["supported_decimals"] for k, v in un["metric_intervals"].items()}
    checks = [
        ("suitability PR-AUC", "%.*f" % (max(1, dp.get("pr_auc", 3)),
                                         su["model_pu"]["pr_auc"])),
        ("certified ratio", "%.2f" % (100 * ce["greedy_ratio"])),
        ("sites always selected", str(un["plan_stability"]["sites_always_selected"])),
    ]
    bad = [(n, v) for n, v in checks if v not in plan]
    for n, v in bad:
        print(f"::error::PLAN.md does not contain current {n} = {v}")
    if bad:
        return 1
    print("PLAN.md figures agree with reports/ (%d checks)" % len(checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
