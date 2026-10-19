"""Independently reconcile final notebook CSVs with the original provider replies."""

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def records(name):
    with (DATA / name).open() as f:
        return [json.loads(line) for line in f]


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    for name, expected in manifest["file_sha256"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == expected, name
    raw = records("responses.jsonl")
    attempts = records("attempts.jsonl")
    assert len(raw) == len({r["request_id"] for r in raw}) == manifest["terminal_calls"]
    provider_ids = [r["provider_response_id"] for r in raw]
    assert all(provider_ids) and len(provider_ids) == len(set(provider_ids))
    by_id = {r["request_id"]: r for r in raw}
    phases = defaultdict(list)
    for row in raw:
        phases[row["phase"]].append(row)
    calibration = phases["calibration"]
    assert len(calibration) == 1000 and all(r["valid"] for r in calibration)
    p = sum(r["value"] == 1 for r in calibration) / 1000
    assert p == manifest["calibration"]["p_hat"]
    cal_ids = {r["request_id"] for r in calibration}
    for row in raw:
        if row["phase"] not in ("integer_pilot", "permutation"):
            mapping = manifest["response_mappings"][row["prompt_id"]]
            parsed = mapping.get(row["raw_response"].strip().lower())
            assert row["value"] == parsed

    clt = csv_rows(DATA / "clt_sample_intermediates.csv")
    assert len(clt) == 2000
    selected_clt_ids = set()
    clt_values = []
    for sample in clt:
        gid = int(float(sample["group_id"]))
        request_ids = [sample[f"request_{i:02d}"] for i in range(40)]
        selected_clt_ids.update(request_ids)
        calls = [by_id[rid] for rid in request_ids]
        assert all(r["valid"] and r["phase"] == "clt" and r["group_id"] == gid and r["position"] == i for i, r in enumerate(calls))
        positive_count = sum(r["value"] == 1 for r in calls)
        # Independently standardize the binomial count rather than the sign sum.
        expected = 10 + 2 * (positive_count - 40*p) / math.sqrt(40*p*(1-p))
        assert math.isclose(float(sample["x"]), expected, abs_tol=1e-12)
        clt_values.append(expected)
    assert len(selected_clt_ids) == 80000 and selected_clt_ids.isdisjoint(cal_ids)

    perm_dir = ROOT / "derived" / "permutation"
    permutation = csv_rows(perm_dir / "normal_samples.csv")
    assert len(permutation) == 2000
    perm_values = []
    perm_ids = set()
    normal = statistics.NormalDist(10, 2)
    for sample in permutation:
        ids = json.loads(sample["request_ids"])
        calls = [by_id[rid] for rid in ids]
        perm_ids.update(ids)
        values = [int(r["raw_response"].strip()) for r in calls]
        assert len(values) == len(set(values)) == 10
        assert all(r["phase"] == "permutation" and r["valid"] and r["group_id"] == int(sample["group_id"]) and r["position"] == i for i, r in enumerate(calls))
        # Sorted-list removal gives lexicographic offsets, independent of the
        # notebook's counts of smaller later elements.
        remaining = sorted(values)
        k = 0
        for value in values:
            offset = remaining.index(value)
            k += offset * math.factorial(len(remaining)-1)
            remaining.remove(value)
        u = (k + .5)/math.factorial(10)
        expected = normal.inv_cdf(u)
        assert Decimal(sample["K"]) == k
        assert math.isclose(float(sample["U"]), u, abs_tol=1e-15)
        assert math.isclose(float(sample["X"]), expected, abs_tol=1e-12)
        perm_values.append(expected)
    assert len(perm_ids) == 20000 and perm_ids.isdisjoint(selected_clt_ids | cal_ids)

    def summary(values):
        return {"n": len(values), "mean": statistics.mean(values),
            "sample_standard_deviation": statistics.stdev(values),
            "min": min(values), "max": max(values)}

    start = min(datetime.fromisoformat(r["started_at_utc"]) for r in raw)
    finish = max(datetime.fromisoformat(r["finished_at_utc"]) for r in raw)
    result = {
        "verified": True, "terminal_calls": len(raw), "http_attempts": len(attempts),
        "http_status_counts": dict(Counter(str(r.get("http_status")) for r in attempts)),
        "phase_counts": {phase: len(rows) for phase, rows in phases.items()},
        "models": dict(Counter(r["model"] for r in raw)),
        "fingerprints": dict(Counter(r.get("system_fingerprint") for r in raw)),
        "total_tokens": sum((r.get("usage") or {}).get("total_tokens", 0) for r in attempts),
        "first_started_utc": start.isoformat(), "last_finished_utc": finish.isoformat(),
        "elapsed_seconds_including_between_phase_analysis": (finish-start).total_seconds(),
        "calibration_p_hat": p,
        "production_p_hat": sum(by_id[rid]["value"] == 1 for rid in selected_clt_ids) / 80000,
        "clt": summary(clt_values), "permutation": summary(perm_values),
        "permutation_total_groups": manifest["phases"]["permutation"]["total_groups"],
        "selected_raw_request_ids_disjoint": True,
        "provider_response_ids_unique": True,
        "csv_transforms_reconciled_to_raw": True,
    }
    (DATA / "final_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
