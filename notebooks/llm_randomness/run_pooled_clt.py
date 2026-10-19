"""Extend an audited historical calibration pool, then collect fresh CLT samples."""

import argparse
import asyncio
import csv
import hashlib
import json
import signal
from pathlib import Path

from run_experiment import Collector, ROOT, spec


class PooledCollector(Collector):
    def __init__(self, args):
        selection_path = Path(args.selection_from).resolve()
        # The base runner's ordinary inherited-selection path is for fresh-only
        # calibration. This wrapper records the explicit historical reuse instead.
        args.selection_from = None
        super().__init__(args)
        args.selection_from = selection_path
        try:
            self._initialize_pool(args, selection_path)
        except BaseException:
            self.responses_file.close()
            self.attempts_file.close()
            self.started_file.close()
            raise

    def _initialize_pool(self, args, selection_path):
        source_bytes = selection_path.read_bytes()
        source = json.loads(source_bytes)
        if source["settings"] != self.manifest["settings"] or source["endpoint"] != self.manifest["endpoint"]:
            raise ValueError("Historical selection model/settings/endpoint do not match")
        selection = source["selection"]
        if self.manifest.get("selection", selection) != selection:
            raise ValueError("Cannot change inherited prompt selection")
        self.manifest["selection"] = selection
        selection_source = {"source_manifest": str(selection_path),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "description": "Prompt inherited; explicitly pooled historical replies are calibration only"}
        if self.manifest.get("selection_source", selection_source) != selection_source:
            raise ValueError("Cannot change inherited selection source")
        self.manifest["selection_source"] = selection_source
        self.manifest["required_phases"] = ["calibration", "clt"]
        self.manifest.pop("pilot_expansion", None)

        summary_path = Path(args.pool_summary).resolve()
        summary_bytes = summary_path.read_bytes()
        summary = json.loads(summary_bytes)
        csv_path = summary_path.parent / "pooled_flips.csv"
        csv_bytes = csv_path.read_bytes()
        csv_hash = hashlib.sha256(csv_bytes).hexdigest()
        if csv_hash != summary["csv_sha256"]:
            raise ValueError("Historical pool CSV hash does not match its summary")
        if summary["settings"] != self.manifest["settings"] or summary["prompt"] != selection["prompt"]:
            raise ValueError("Historical pool settings/prompt differ from current selection")
        mapping = self.manifest["response_mappings"][selection["prompt_id"]]
        if summary["mapping"] != mapping:
            raise ValueError("Historical pool response mapping differs")
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != summary["n"] or len(rows) >= self.calibration_calls:
            raise ValueError("Pool count is inconsistent or leaves no fresh calibration quota")
        if sum(int(r["value"]) == 1 for r in rows) != summary["plus_count"]:
            raise ValueError("Historical pool positive count differs from summary")
        provider_ids = [r["provider_response_id"] for r in rows]
        if not all(provider_ids) or len(provider_ids) != len(set(provider_ids)):
            raise ValueError("Historical pool contains missing or repeated provider response IDs")
        source_order = {r["source_run"]: i for i, r in enumerate(summary["sources"])}
        rows.sort(key=lambda r: (source_order[r["source_run"]],
            0 if r["original_phase"] == "calibration" else 1,
            int(r["request_id"].rsplit(":", 1)[1])))
        provenance = {"source_summary": str(summary_path),
            "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "csv_sha256": csv_hash, "imported_count": len(rows),
            "new_calibration_calls": self.calibration_calls-len(rows),
            "csv_snapshot": "calibration_pool_snapshot.csv",
            "summary_snapshot": "calibration_pool_summary.json"}
        if self.manifest.get("calibration_pool", provenance) != provenance:
            raise ValueError("Cannot change calibration pool on resume")
        self.manifest["calibration_pool"] = provenance
        for name, content in [("calibration_pool_snapshot.csv", csv_bytes),
                              ("calibration_pool_summary.json", summary_bytes)]:
            destination = self.directory/name
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError("Saved historical pool snapshot has changed")
            if not destination.exists():
                destination.write_bytes(content)
        for index, row in enumerate(rows):
            if mapping.get(row["raw_response"].strip().lower()) != int(row["value"]):
                raise ValueError("Historical raw response and mapped value disagree")
            item = spec("calibration", selection["prompt_id"], selection["prompt"], index)
            record = {**item, "record_type": "imported_calibration",
                "calibration_source_run": row["source_run"],
                "original_phase": row["original_phase"], "original_request_id": row["request_id"],
                "provider_response_id": row["provider_response_id"],
                "raw_response": row["raw_response"], "value": int(row["value"]),
                "valid": True, "status": "imported", "model": row["model"],
                "system_fingerprint": row["system_fingerprint"] or None,
                "started_at_utc": row["started_at_utc"], "finished_at_utc": row["finished_at_utc"],
                "elapsed_seconds": None, "usage": None}
            existing = self.responses.get(item["request_id"])
            if existing is not None and existing != record:
                raise ValueError("Historical seed collides with an existing record")
            if existing is None:
                self.responses_file.write(json.dumps(record)+"\n")
                self.responses[item["request_id"]] = record
        self.initial_count = len(self.responses)
        self.manifest["imported_calibration_records"] = len(rows)
        self.manifest["new_api_calls_planned"] = self.calibration_calls-len(rows) + self.flips_per_sample*self.requested_samples
        self.manifest["calibration_time_order"] = "Historical source/run and original planned order, followed by fresh calibration planned order"
        self.write_manifest()
        print(json.dumps({"imported_calibration": len(rows),
            "fresh_calibration_planned": self.calibration_calls-len(rows),
            "fresh_production_planned": self.flips_per_sample*self.requested_samples}), flush=True)

    async def run(self):
        await super().run()
        for filename in ("calibration_pool_snapshot.csv", "calibration_pool_summary.json"):
            self.manifest["file_sha256"][filename] = hashlib.sha256((self.directory/filename).read_bytes()).hexdigest()
        self.manifest["new_api_responses"] = len(self.responses)-self.manifest["imported_calibration_records"]
        self.write_manifest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"new_pooled_run")
    parser.add_argument("--config", default="provider.toml")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--calibration-calls", type=int, default=160000)
    parser.add_argument("--flips-per-sample", type=int, default=40)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--selection-from", type=Path, default=ROOT/"data/manifest.json")
    parser.add_argument("--pool-summary", type=Path, default=ROOT/"calibration_pool/summary.json")
    args = parser.parse_args()
    args.phase = "clt_only"
    if min(args.concurrency, args.calibration_calls, args.flips_per_sample, args.samples) < 1:
        parser.error("Counts and concurrency must be positive")
    collector = PooledCollector(args)
    def stop_requested(signum, frame):
        raise KeyboardInterrupt("Stopped by user")
    signal.signal(signal.SIGTERM, stop_requested)
    try:
        asyncio.run(collector.run())
    except BaseException as exc:
        collector.manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        collector.manifest["last_error"] = str(exc).replace(collector.key, "[REDACTED]")
        collector.write_manifest()
        raise
    finally:
        collector.responses_file.close()
        collector.attempts_file.close()
        collector.started_file.close()


if __name__ == "__main__":
    main()
