"""End-to-end pipeline runner.

    python run_pipeline.py            # full run
    python run_pipeline.py --from features
    python run_pipeline.py --only export

Stages are ordered and idempotent; each writes its artefacts to data/ and
reports/ so a later stage can be re-run alone after a parameter change.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

STAGES = [
    ("stations", "evopt.ingest.stations", "AFDC / Germany / Connecticut -> parquet"),
    ("mobility", "evopt.ingest.mobility", "47.4M taxi trips -> H3 demand surface"),
    ("traffic", "evopt.ingest.traffic", "3.3 GB counts -> H3 throughput + morphology"),
    ("landuse", "evopt.ingest.landuse", "858k PLUTO tax lots -> built-environment features"),
    ("acs", "evopt.ingest.acs", "ACS B25044 vehicle availability -> H3"),
    ("features", "evopt.features", "join + spatial lag + leave-one-cell-out supply"),
    ("ghsl", "evopt.ingest.ghsl", "GHSL built form + population (global, for transfer)"),
    ("feeders", "evopt.ingest.feeders", "Con Ed substation headroom -> H3"),
    ("surface", "evopt.surface", "composite demand, conserved capacity, gap"),
    ("demand", "evopt.models.demand", "Tweedie LGBM + spatial block CV"),
    ("suitability", "evopt.models.suitability", "bagged PU + forward temporal test"),
    ("allocate", "evopt.models.allocate", "capacitated MCLP + M/M/c sizing"),
    ("certify", "evopt.models.certify", "MILP optimum vs greedy (CBC)"),
    ("explain", "evopt.models.explain", "TreeSHAP attribution + equity counterfactual"),
    ("uncertainty", "evopt.models.uncertainty", "bag bootstrap + seed sweep"),
    ("transfer", "evopt.models.transfer", "DE / NYC / CT cross-domain validation"),
    ("export", "evopt.export", "dashboard payload"),
    ("drift", "evopt.drift", "source freshness + metric control limits"),
]


def main() -> int:
    names = [s[0] for s in STAGES]
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=names)
    ap.add_argument("--only", dest="only", choices=names)
    a = ap.parse_args()

    todo = STAGES
    if a.only:
        todo = [s for s in STAGES if s[0] == a.only]
    elif a.start:
        todo = STAGES[names.index(a.start):]

    import runpy
    t0 = time.time()
    for i, (name, mod, desc) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {name} — {desc}")
        print("-" * 68)
        t = time.time()
        runpy.run_module(mod, run_name="__main__")
        print(f"-- {name} done in {time.time() - t:.1f}s")
    print(f"\nPipeline complete in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
