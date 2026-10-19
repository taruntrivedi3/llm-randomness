"""Offline pooled-calibration audit. All requests use httpx.MockTransport."""

import csv
import hashlib
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

import run_experiment as runner
from run_pooled_clt import PooledCollector


class PooledCollectorAudit(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pooled_clt_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pool = self.root / "pool"
        self.pool.mkdir()
        self.config = self.root / "config.toml"
        self.config.write_text('model_provider="mock"\n[model_providers.mock]\nbase_url="https://mock.invalid/v1"\nenv_key="POOLED_CLT_TEST_KEY"\n')
        environment = patch.dict(os.environ, {"POOLED_CLT_TEST_KEY": "test-only-key"})
        environment.start()
        self.addCleanup(environment.stop)
        self.settings = {"model": "mock-model", "temperature": 1.0, "top_p": 1.0, "max_tokens": 32, "n": 1, "stream": False}
        self.selection = {"prompt_id": "binary_11", "prompt": runner.PROMPTS[11]}
        self.source = self.root / "previous_manifest.json"
        self.source.write_text(json.dumps({"settings": self.settings, "endpoint": "https://mock.invalid/v1/chat/completions", "selection": self.selection}))
        self.rows = []
        # Deliberately shuffled original phases/indices; the seed order is audited.
        for index, (phase, original_index, raw) in enumerate([
            ("clt", 0, " B\n"), ("calibration", 1, "A"),
            ("calibration", 0, "B"), ("clt", 1, "A"),
        ]):
            self.rows.append({
                "source_run": "previous", "original_phase": phase,
                "request_id": f"{phase}:binary_11:{original_index:08d}",
                "provider_response_id": f"old-{index}", "raw_response": raw,
                "value": 1 if raw.strip() == "B" else -1, "model": "mock-model",
                "system_fingerprint": "old-fingerprint", "started_at_utc": "2026-01-01T00:00:00+00:00",
                "finished_at_utc": "2026-01-01T00:00:01+00:00",
            })
        self.write_pool()
        self.args = SimpleNamespace(
            config=str(self.config), data_dir=self.root / "new_data", model="mock-model",
            concurrency=4, calibration_calls=10, flips_per_sample=3, samples=2,
            selection_from=self.source, pool_summary=self.pool / "summary.json", phase="clt_only",
        )

    def write_pool(self):
        path = self.pool / "pooled_flips.csv"
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        summary = {
            "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n": len(self.rows),
            "plus_count": sum(row["value"] == 1 for row in self.rows), "settings": self.settings,
            "prompt": self.selection["prompt"], "mapping": {"a": -1, "b": 1},
            "sources": [{"source_run": "previous"}],
        }
        (self.pool / "summary.json").write_text(json.dumps(summary))

    def collector(self, **overrides):
        args = SimpleNamespace(**{**vars(self.args), **overrides})
        with redirect_stdout(io.StringIO()):
            collector = PooledCollector(args)
        self.addCleanup(self.close, collector)
        return collector

    @staticmethod
    def close(collector):
        for name in ("responses_file", "attempts_file", "started_file"):
            getattr(collector, name).close()

    @staticmethod
    def reply(raw, response_id):
        return httpx.Response(200, json={"id": response_id, "model": "mock-model",
            "choices": [{"message": {"content": raw}, "finish_reason": "stop"}]})

    def records(self, filename):
        return runner.read_jsonl(self.args.data_dir / filename)

    async def test_seed_once_then_six_fresh_calibration_and_six_fresh_production_calls(self):
        collector = self.collector()
        imported = self.records("responses.jsonl")
        self.assertEqual(len(imported), 4)
        self.assertEqual([row["provider_response_id"] for row in imported], ["old-2", "old-1", "old-0", "old-3"])
        self.assertEqual([row["planned_index"] for row in imported], [0, 1, 2, 3])
        self.assertEqual(self.records("requests_started.jsonl"), [])
        self.assertEqual(self.records("attempts.jsonl"), [])
        self.assertEqual(collector.manifest["selection_source"]["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(collector.manifest["new_api_calls_planned"], 12)
        self.close(collector)
        collector = self.collector()
        self.assertEqual(len(self.records("responses.jsonl")), 4)
        sent = []
        # Pool: 2 plus / 4; fresh calibration: 4 plus / 6 => 6 plus / 10.
        answers = iter(["B"] * 4 + ["A"] * 2 + ["A", "B", "A", "B", "B", "B"])

        def handler(request):
            sent.append(request)
            return self.reply(next(answers), f"new-{len(sent)}")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch.object(runner.httpx, "AsyncClient", return_value=client), redirect_stdout(io.StringIO()):
            await collector.run()
        self.assertEqual(len(sent), 12)
        rows = self.records("responses.jsonl")
        calibration = [row for row in rows if row["phase"] == "calibration"]
        production = [row for row in rows if row["phase"] == "clt"]
        self.assertEqual(len(calibration), 10)
        self.assertEqual(len(production), 6)
        self.assertEqual(sum(row.get("record_type") == "imported_calibration" for row in rows), 4)
        self.assertTrue(all(row["provider_response_id"].startswith("new-") for row in production))
        self.assertEqual(sorted(row["planned_index"] for row in calibration), list(range(10)))
        self.assertEqual(sorted(row["planned_index"] for row in production), list(range(6)))
        self.assertEqual({row["group_id"] for row in production}, {0, 1})
        self.assertEqual(collector.manifest["calibration"]["p_hat"], .6)
        self.assertAlmostEqual(collector.manifest["calibration"]["mu"], .2)
        self.assertAlmostEqual(collector.manifest["calibration"]["sigma"], .9797958971132712)
        self.assertAlmostEqual(collector.manifest["calibration"]["target_mean_calibration_standard_error_approx"], 1.0954451150103321)
        self.assertEqual(collector.manifest["status"], "complete")
        self.assertEqual(collector.manifest["new_api_responses"], 12)
        self.assertEqual(len(self.records("requests_started.jsonl")), 12)
        self.assertEqual(len(self.records("attempts.jsonl")), 12)
        self.assertEqual(collector.manifest["file_sha256"]["calibration_pool_snapshot.csv"], hashlib.sha256((self.pool / "pooled_flips.csv").read_bytes()).hexdigest())
        self.close(collector)
        collector = self.collector()
        self.assertEqual(len(self.records("responses.jsonl")), 16)
        self.assertEqual(collector.groups("clt", 3, False)[0], [0, 1])

    async def test_partial_calibration_resume_only_sends_missing_quota(self):
        collector = self.collector()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: self.reply("B", "fresh-before-interruption"))) as client:
            collector.client = client
            await collector.call(runner.spec("calibration", "binary_11", runner.PROMPTS[11], 4))
        self.close(collector)
        collector = self.collector()
        sent = []

        def handler(request):
            sent.append(request)
            return self.reply("A" if len(sent) <= 2 else "B", f"after-resume-{len(sent)}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            collector.client = client
            with redirect_stdout(io.StringIO()):
                await collector.calibration()
            await collector.production("clt")
        self.assertEqual(len(sent), 11)
        self.assertEqual(len(self.records("requests_started.jsonl")), 12)
        rows = self.records("responses.jsonl")
        self.assertEqual(len(rows), 16)
        self.assertEqual(sum(row["provider_response_id"] == "fresh-before-interruption" for row in rows), 1)
        self.assertEqual(sum(row.get("record_type") == "imported_calibration" for row in rows), 4)
        self.assertEqual(collector.groups("clt", 3, False)[0], [0, 1])

    def test_changed_pool_is_rejected_on_resume_even_with_updated_valid_summary_hash(self):
        collector = self.collector()
        original_records = (self.args.data_dir / "responses.jsonl").read_bytes()
        self.close(collector)
        self.rows[0]["raw_response"] = "A"
        self.rows[0]["value"] = -1
        self.write_pool()
        with self.assertRaisesRegex(ValueError, "Cannot change calibration pool"):
            self.collector()
        self.assertEqual((self.args.data_dir / "responses.jsonl").read_bytes(), original_records)

    def test_changed_source_and_calibration_count_are_rejected(self):
        collector = self.collector()
        self.close(collector)
        with self.assertRaisesRegex(ValueError, "changed calibration_calls"):
            self.collector(calibration_calls=11)
        self.source.write_text(self.source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Cannot change inherited selection source"):
            self.collector()

    def test_duplicate_historical_provider_ids_are_rejected_before_seed_import(self):
        self.rows[1]["provider_response_id"] = self.rows[0]["provider_response_id"]
        self.write_pool()
        with self.assertRaisesRegex(ValueError, "repeated provider response IDs"):
            self.collector()
        self.assertEqual(self.records("responses.jsonl"), [])
        self.assertEqual(self.records("requests_started.jsonl"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
