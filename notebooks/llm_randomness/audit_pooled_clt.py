"""Reconcile a completed pooled-calibration notebook against original replies."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
        default=Path(__file__).resolve().parent/"runs/clt_40_cal160000/data")
    directory = parser.parse_args().data_dir
    manifest = json.loads((directory/"manifest.json").read_text())
    assert manifest["status"] == "complete"
    for name, expected in manifest["file_sha256"].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected, name
    with (directory/"responses.jsonl").open() as f:
        replies = [json.loads(line) for line in f]
    by_id = {r["request_id"]: r for r in replies}
    assert len(replies) == len(by_id) == manifest["terminal_calls"]
    ids = [r["provider_response_id"] for r in replies]
    assert all(ids) and len(ids) == len(set(ids))
    calibration = [r for r in replies if r["phase"] == "calibration"]
    production = [r for r in replies if r["phase"] == "clt"]
    historical = [r for r in calibration if r.get("record_type") == "imported_calibration"]
    fresh_calibration = [r for r in calibration if r.get("record_type") != "imported_calibration"]
    assert len(calibration) == manifest["calibration_calls"]
    assert len(historical) == manifest["calibration_pool"]["imported_count"]
    assert len(fresh_calibration) == manifest["calibration_pool"]["new_calibration_calls"]
    assert all(r["valid"] for r in calibration)
    assert all(r.get("record_type") != "imported_calibration" for r in production)
    calibration_end = max(datetime.fromisoformat(r["finished_at_utc"]) for r in fresh_calibration)
    production_start = min(datetime.fromisoformat(r["started_at_utc"]) for r in production)
    assert calibration_end <= production_start, "Production began before calibration finished"
    for row in replies:
        mapping = manifest["response_mappings"][row["prompt_id"]]
        raw = row.get("raw_response")
        parsed = mapping.get(raw.strip().lower()) if isinstance(raw, str) else None
        assert row["value"] == parsed
    pool_path = directory/manifest["calibration_pool"]["csv_snapshot"]
    with pool_path.open() as f:
        original_pool = {row["provider_response_id"]: row for row in csv.DictReader(f)}
    assert len(original_pool) == len(historical)
    for row in historical:
        source = original_pool[row["provider_response_id"]]
        assert row["raw_response"] == source["raw_response"]
        assert row["value"] == int(source["value"])
        assert row["original_request_id"] == source["request_id"]
        assert row["original_phase"] == source["original_phase"]
        assert row["calibration_source_run"] == source["source_run"]
    p = sum(r["value"] == 1 for r in calibration)/len(calibration)
    assert p == manifest["calibration"]["p_hat"]
    n = manifest["flips_per_sample"]
    with (directory/"clt_sample_intermediates.csv").open() as f:
        samples = list(csv.DictReader(f))
    assert len(samples) == manifest["requested_samples_per_method"]
    used = set()
    values = []
    group_ids = []
    for sample in samples:
        gid = int(float(sample["group_id"]))
        group_ids.append(gid)
        request_ids = [sample[f"request_{i:02d}"] for i in range(n)]
        assert used.isdisjoint(request_ids)
        used.update(request_ids)
        raw = [by_id[rid] for rid in request_ids]
        assert all(r["phase"] == "clt" and r["group_id"] == gid and r["position"] == i
            and r["planned_index"] == n*gid+i and r["valid"] for i,r in enumerate(raw))
        positive_count = sum(r["value"] == 1 for r in raw)
        expected = 10 + 2*(positive_count-n*p)/math.sqrt(n*p*(1-p))
        assert math.isclose(float(sample["x"]), expected, abs_tol=1e-12)
        values.append(expected)
    assert group_ids == manifest["phases"]["clt"]["selected_group_ids"]
    assert len(used) == n*len(samples)
    assert used.isdisjoint(r["request_id"] for r in calibration)
    production_p = sum(by_id[rid]["value"] == 1 for rid in used)/len(used)
    source_summary = []
    for label, group in [("historical_pool",historical),("new_calibration",fresh_calibration),
                          ("all_calibration",calibration),("new_production",production)]:
        valid = [r for r in group if r["valid"]]
        source_summary.append({"source":label,"records":len(group),"valid":len(valid),
            "plus_count":sum(r["value"] == 1 for r in valid),
            "p_hat":sum(r["value"] == 1 for r in valid)/len(valid)})
    attempt_counts = Counter()
    attempts = total_tokens = http_successes = 0
    for line in (directory/"attempts.jsonl").open():
        row = json.loads(line)
        attempts += 1
        attempt_counts[f"{row['status']}:{row.get('http_status')}"] += 1
        total_tokens += (row.get("usage") or {}).get("total_tokens", 0)
        http_successes += row.get("http_status") == 200
    fresh = fresh_calibration+production
    assert len(fresh) == http_successes == manifest["new_api_responses"]
    beginning = min(datetime.fromisoformat(r["started_at_utc"]) for r in fresh)
    end = max(datetime.fromisoformat(r["finished_at_utc"]) for r in fresh)
    iid_sd = 2*math.sqrt(production_p*(1-production_p)/(p*(1-p)))
    summary = {"verified":True,"calibration_records":len(calibration),
        "historical_calibration_records":len(historical),"new_calibration_calls":len(fresh_calibration),
        "new_production_calls":len(production),"new_api_responses":len(fresh),
        "http_attempts":attempts,"attempt_status_counts":dict(attempt_counts),
        "total_tokens_new_calls":total_tokens,"sources":source_summary,
        "samples":len(values),"flips_per_sample":n,"mean":statistics.mean(values),
        "sample_standard_deviation":statistics.stdev(values),
        "iid_standard_deviation_at_observed_production_p":iid_sd,
        "variance_ratio_to_iid_binomial":statistics.variance(values)/(iid_sd*iid_sd),
        "min":min(values),"max":max(values),
        "calibration_p_hat":p,"production_p_hat":production_p,
        "first_new_call_utc":beginning.isoformat(),"last_new_call_utc":end.isoformat(),
        "elapsed_seconds_new_calls":(end-beginning).total_seconds(),
        "models":dict(Counter(r["model"] for r in fresh)),
        "fingerprints_new_calls":dict(Counter(r.get("system_fingerprint") for r in fresh)),
        "no_calibration_input_reused_in_production":True,
        "production_started_after_calibration_finished":True,
        "all_provider_response_ids_unique":True,"exports_match_raw_replies":True}
    (directory/"final_audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
