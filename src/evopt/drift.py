"""Deployment - freshness and drift checks.

Two questions a deployed pipeline has to answer without a human watching it.

(1) Is anything stale?
    Raw sources change: AFDC republishes, PLUTO gets a new release, Con Edison
    refreshes hosting capacity twice a year. A manifest of source fingerprints
    says which stages need re-running rather than re-running all seventeen.

(2) Has the model drifted, or is this just noise?
    This is the question that needs Stage 7g. PR-AUC moves +/-0.003 between
    identical runs because LightGBM's threaded histogram build is not
    bit-deterministic, so a bare threshold ("alert if PR-AUC drops 0.01") would
    fire on nothing at all. The seed sweep already established the 95% interval
    the metric occupies under no change, so **that interval is the control
    limit**: inside it is noise, outside it is drift worth a human.

    The temporal split needs the same treatment. The forward window is fixed at
    2019-2021, but AFDC keeps publishing. Once enough stations have opened past
    the window, the split is measuring history rather than the present and the
    cutoff should advance -- so the count of post-window records is reported
    alongside the metric check.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from .config import (RAW, REPORTS, AFDC_NATIONAL, TRAIN_CUTOFF, TEST_WINDOW)

MANIFEST = REPORTS / "manifest.json"

# Which stages depend on which raw sources. Re-running a source's dependents is
# cheaper and more honest than re-running everything.
SOURCE_STAGES = {
    "stations/afdc_national": ["stations", "features", "surface", "suitability",
                               "allocate", "certify", "explain", "uncertainty",
                               "transfer", "export"],
    "stations/germany": ["stations", "transfer"],
    "mobility/nyc_taxi": ["mobility", "features", "surface", "demand",
                          "suitability", "allocate", "export"],
    "traffic/nyc_volume": ["traffic", "features", "surface", "demand",
                           "suitability", "allocate", "export"],
    "landuse": ["landuse", "acs", "features", "surface", "demand",
                "suitability", "allocate", "export"],
    "acs": ["acs", "features", "surface", "demand", "suitability", "export"],
    "grid": ["feeders", "allocate", "certify", "export"],
    "ghsl": ["ghsl", "transfer"],
}

# A large CSV is fingerprinted by size plus head/tail bytes rather than a full
# hash: a 3.3 GB file would otherwise take minutes to check, and any real
# republication changes the size or the first rows.
HEAD_BYTES = 1 << 20


def fingerprint(path) -> str:
    st = path.stat()
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(HEAD_BYTES))
        if st.st_size > HEAD_BYTES * 2:
            f.seek(-HEAD_BYTES, 2)
            h.update(f.read(HEAD_BYTES))
    return h.hexdigest()[:16]


def scan_sources() -> dict:
    out = {}
    for group in SOURCE_STAGES:
        d = RAW / group
        if not d.exists():
            continue
        files = sorted(p for p in d.rglob("*")
                       if p.is_file() and p.suffix in (".csv", ".parquet", ".tif"))
        out[group] = {p.name: fingerprint(p) for p in files}
    return out


def temporal_status() -> dict:
    """Is the fixed forward window still the present, or now history?"""
    d = pd.read_csv(AFDC_NATIONAL, usecols=["Fuel Type Code", "Open Date"],
                    low_memory=False)
    d = d[d["Fuel Type Code"] == "ELEC"]
    od = pd.to_datetime(d["Open Date"], errors="coerce").dropna()
    w0, w1 = map(pd.Timestamp, TEST_WINDOW)
    after = int((od > w1).sum())
    return {
        "train_cutoff": TRAIN_CUTOFF,
        "forward_window": list(TEST_WINDOW),
        "latest_open_date": str(od.max().date()) if len(od) else None,
        "stations_after_window": after,
        "share_after_window": round(float(after / max(len(od), 1)), 4),
        # A window that no longer contains the newest build-out is measuring
        # history. 5% is a deliberately conservative trigger for a human to
        # re-cut the split rather than an automatic action.
        "split_should_advance": bool(after / max(len(od), 1) > 0.05),
    }


def metric_status() -> dict:
    """Compare the current forward-test metrics against 7g's control limits."""
    su = REPORTS / "suitability_metrics.json"
    un = REPORTS / "uncertainty.json"
    if not (su.exists() and un.exists()):
        return {"available": False}
    cur = json.loads(su.read_text())["model_pu"]
    iv = json.loads(un.read_text())["metric_intervals"]

    checks = {}
    for k in ("pr_auc", "precision@100", "lift@100", "ndcg"):
        if k not in iv or k not in cur:
            continue
        lo, hi = iv[k]["lo95"], iv[k]["hi95"]
        v = cur[k]
        checks[k] = {"value": round(v, 5), "lo95": round(lo, 5),
                     "hi95": round(hi, 5),
                     "within_control_limits": bool(lo <= v <= hi)}
    drifted = [k for k, c in checks.items() if not c["within_control_limits"]]
    return {"available": True, "checks": checks, "drifted": drifted,
            "drift_detected": bool(drifted)}


def run(update: bool = False) -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    current = scan_sources()
    prev = json.loads(MANIFEST.read_text()).get("sources", {}) \
        if MANIFEST.exists() else {}

    changed_groups, new_groups = [], []
    for g, files in current.items():
        if g not in prev:
            new_groups.append(g)
        elif prev[g] != files:
            changed_groups.append(g)

    stale = sorted({s for g in changed_groups + new_groups
                    for s in SOURCE_STAGES.get(g, [])})

    res = {
        "checked": now,
        "sources_tracked": sum(len(v) for v in current.values()),
        "groups_changed": changed_groups,
        "groups_new": new_groups,
        "stages_to_rerun": stale if prev else ["<no baseline: run with --update>"],
        "temporal": temporal_status(),
        "metrics": metric_status(),
    }
    res["action_required"] = bool(
        changed_groups or res["temporal"]["split_should_advance"]
        or res["metrics"].get("drift_detected"))

    (REPORTS / "drift.json").write_text(json.dumps(res, indent=2))
    if update or not MANIFEST.exists():
        MANIFEST.write_text(json.dumps({"updated": now, "sources": current},
                                       indent=2))
    return res


if __name__ == "__main__":
    import sys
    r = run(update="--update" in sys.argv)
    print("Deployment - freshness and drift")
    print("  sources tracked : %d files across %d groups"
          % (r["sources_tracked"], len(SOURCE_STAGES)))
    print("  changed         : %s" % (r["groups_changed"] or "none"))
    if r["groups_new"]:
        print("  newly tracked   : %s" % r["groups_new"])
    print("  stages to re-run: %s" % (r["stages_to_rerun"] or "none"))

    t = r["temporal"]
    print("\n  TEMPORAL SPLIT")
    print("    cutoff %s, forward window %s to %s"
          % (t["train_cutoff"], t["forward_window"][0], t["forward_window"][1]))
    print("    latest AFDC open date : %s" % t["latest_open_date"])
    print("    stations after window : %s (%.1f%%)  -> advance split: %s"
          % (f'{t["stations_after_window"]:,}', 100 * t["share_after_window"],
             "YES" if t["split_should_advance"] else "no"))

    m = r["metrics"]
    if m.get("available"):
        print("\n  METRIC CONTROL LIMITS (Stage 7g 95% interval)")
        for k, c in m["checks"].items():
            print("    %-14s %8.4f  [%.4f, %.4f]  %s"
                  % (k, c["value"], c["lo95"], c["hi95"],
                     "ok" if c["within_control_limits"] else "DRIFT"))
    print("\n  action required : %s" % ("YES" if r["action_required"] else "no"))
