"""Offline audit of collection and resume behavior; never uses a live endpoint.

Run with an environment containing httpx:
    python notebooks/llm_randomness/test_runner.py
"""

import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

import run_experiment as runner


class CollectorAudit(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="llm_randomness_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / "provider.toml"
        self.write_config("https://mock.invalid/v1")
        environment = patch.dict(os.environ, {"LLM_RANDOMNESS_TEST_KEY": "test-only-key"})
        environment.start()
        self.addCleanup(environment.stop)
        self.args = SimpleNamespace(
            config=str(self.config), data_dir=self.root / "data",
            model="mock-model", concurrency=30,
        )

    def write_config(self, endpoint):
        self.config.write_text(
            'model_provider="offline"\n[model_providers.offline]\n'
            f'base_url="{endpoint}"\nenv_key="LLM_RANDOMNESS_TEST_KEY"\n'
        )

    def collector(self):
        collector = runner.Collector(self.args)
        self.addCleanup(self.close, collector)
        return collector

    @staticmethod
    def close(collector):
        for name in ("responses_file", "attempts_file", "started_file"):
            getattr(collector, name).close()

    @staticmethod
    def reply(raw, response_id="mock-response", finish_reason="stop"):
        return httpx.Response(200, json={
            "id": response_id, "model": "mock-model", "system_fingerprint": "mock-fingerprint",
            "choices": [{"message": {"content": raw}, "finish_reason": finish_reason}],
        })

    @staticmethod
    def flip(index=0):
        return runner.spec("clt", "binary_11", runner.PROMPTS[11], index, 40)

    def records(self, filename):
        return runner.read_jsonl(self.args.data_dir / filename)

    async def test_grouping_uses_planned_order_and_excludes_whole_invalid_or_tied_groups(self):
        collector = self.collector()
        gates = [asyncio.Event() for _ in range(30)]
        all_started = asyncio.Event()
        requests = []

        async def handler(request):
            index = len(requests)
            requests.append(json.loads(request.content))
            if len(requests) == 30:
                all_started.set()
            await gates[index].wait()
            raw = "not-an-integer" if index == 13 else str(index if index < 20 else 42)
            return self.reply(raw, f"mock-{index}")

        planned = [runner.spec("permutation", "integer_0", runner.INTEGER_PROMPT, i, 10) for i in range(30)]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            collector.client = client
            task = asyncio.create_task(collector.collect(planned))
            await asyncio.wait_for(all_started.wait(), timeout=2)
            for gate in reversed(gates):
                gate.set()
                await asyncio.sleep(0)
            await task
        rows = self.records("responses.jsonl")
        self.assertEqual(rows[0]["planned_index"], 29)
        self.assertEqual(len({row["request_id"] for row in rows}), 30)
        accepted, groups = collector.groups("permutation", 10, True)
        self.assertEqual(accepted, [0])
        self.assertEqual([len(groups[i]) for i in range(3)], [10, 10, 10])
        self.assertEqual([row["value"] for row in sorted(groups[0], key=lambda r: r["position"])], list(range(10)))
        self.assertTrue(all(len(request["messages"]) == 1 for request in requests))

    async def test_label_mapping_preserves_raw_content_and_rejects_extra_text(self):
        collector = self.collector()
        answers = iter([" B\n", "a", "A or B", "1"])
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: self.reply(next(answers)))) as client:
            collector.client = client
            for index in range(4):
                await collector.call(self.flip(index))
        rows = self.records("responses.jsonl")
        self.assertEqual([row["value"] for row in rows], [1, -1, None, None])
        self.assertEqual([row["valid"] for row in rows], [True, True, False, False])
        self.assertEqual(rows[0]["raw_response"], " B\n")
        self.assertEqual(rows[0]["system_fingerprint"], "mock-fingerprint")
        self.assertEqual(collector.groups("clt", 40, False)[0], [])

    async def test_resume_finishes_partial_group_without_repeating_completed_calls(self):
        collector = self.collector()
        planned = [runner.spec("permutation", "integer_0", runner.INTEGER_PROMPT, i, 10) for i in range(10)]
        draws = iter(range(10))
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: self.reply(str(next(draws))))) as client:
            collector.client = client
            await collector.collect(planned[:5])
        self.close(collector)
        collector = self.collector()
        sent = []

        def handler(request):
            sent.append(request)
            return self.reply(str(next(draws)))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            collector.client = client
            await collector.collect(planned)
            await collector.collect(planned)
        self.assertEqual(len(sent), 5)
        self.assertEqual(len(self.records("responses.jsonl")), 10)
        self.assertEqual(collector.groups("permutation", 10, True)[0], [0])

    async def test_transient_failure_retries_and_logs_every_attempt(self):
        collector = self.collector()
        replies = iter([httpx.Response(429, text="rate limited"), self.reply("B")])
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(replies))) as client:
            collector.client = client
            with patch.object(runner.asyncio, "sleep", new_callable=AsyncMock):
                await collector.call(self.flip())
        self.assertEqual([row["status"] for row in self.records("attempts.jsonl")], ["http_error", "ok"])
        self.assertEqual([row["attempt"] for row in self.records("requests_started.jsonl")], [1, 2])
        self.assertEqual(len(self.records("responses.jsonl")), 1)

    async def test_malformed_success_is_logged_before_retry(self):
        collector = self.collector()
        malformed = [None, {"choices": None}, {"choices": [{"message": None}]}]
        replies = iter([httpx.Response(200, json=value) for value in malformed] + [self.reply("A")])
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(replies))) as client:
            collector.client = client
            with patch.object(runner.asyncio, "sleep", new_callable=AsyncMock):
                await collector.call(self.flip())
        self.assertEqual([row["status"] for row in self.records("attempts.jsonl")], ["transport_error"] * 3 + ["ok"])
        self.assertEqual(len(self.records("requests_started.jsonl")), 4)
        self.assertEqual(len(self.records("responses.jsonl")), 1)

    async def test_permanent_http_failure_stops_without_fake_terminal_response(self):
        collector = self.collector()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(401, text="test-only-key rejected"))) as client:
            collector.client = client
            with self.assertRaisesRegex(RuntimeError, "Non-retryable HTTP 401"):
                await collector.collect([self.flip()])
        self.assertEqual(len(self.records("attempts.jsonl")), 1)
        self.assertEqual(self.records("responses.jsonl"), [])
        self.assertNotIn("test-only-key", (self.args.data_dir / "attempts.jsonl").read_text())

    async def test_cancelled_sent_request_is_journaled_and_resume_increments_attempt(self):
        collector = self.collector()
        sent = asyncio.Event()

        async def hanging(request):
            sent.set()
            await asyncio.Event().wait()

        async with httpx.AsyncClient(transport=httpx.MockTransport(hanging)) as client:
            collector.client = client
            task = asyncio.create_task(collector.call(self.flip()))
            await asyncio.wait_for(sent.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(len(self.records("requests_started.jsonl")), 1)
        self.assertEqual(self.records("responses.jsonl"), [])
        self.close(collector)
        collector = self.collector()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: self.reply("B"))) as client:
            collector.client = client
            await collector.call(self.flip())
        self.assertEqual([row["attempt"] for row in self.records("requests_started.jsonl")], [1, 2])
        self.assertEqual(self.records("responses.jsonl")[0]["attempt"], 2)

    async def test_resume_recovers_first_durable_terminal_attempt_without_new_draw(self):
        collector = self.collector()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: self.reply("A", "original-response"))) as client:
            collector.client = client
            await collector.call(self.flip())
        self.close(collector)
        # Simulate a crash after attempts append, before responses append.
        (self.args.data_dir / "responses.jsonl").write_text("")
        collector = self.collector()

        def never_send(request):
            self.fail("Durably recorded terminal response must not trigger another draw")

        async with httpx.AsyncClient(transport=httpx.MockTransport(never_send)) as client:
            collector.client = client
            await collector.collect([self.flip()])
        rows = self.records("responses.jsonl")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["provider_response_id"], "original-response")
        self.assertEqual(rows[0]["value"], -1)
        self.assertTrue(rows[0]["recovered_from_attempt_journal"])

    async def test_resume_rejects_changed_spec_before_reusing_response(self):
        collector = self.collector()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: self.reply("B"))) as client:
            collector.client = client
            await collector.call(self.flip())
        for field, value in [("prompt", "Different prompt"), ("group_id", 7), ("position", 4)]:
            with self.subTest(field=field):
                changed = {**self.flip(), field: value}
                with self.assertRaisesRegex(ValueError, "different prompt or grouping"):
                    await collector.call(changed)
                with self.assertRaisesRegex(ValueError, "different prompt or grouping"):
                    await collector.collect([changed])

    def test_resume_rejects_changed_endpoint_or_model(self):
        collector = self.collector()
        self.close(collector)
        self.write_config("https://different.invalid/v1")
        with self.assertRaisesRegex(ValueError, "provider endpoint"):
            self.collector()
        self.write_config("https://mock.invalid/v1")
        self.args.model = "different-model"
        with self.assertRaisesRegex(ValueError, "sampling settings"):
            self.collector()

    def test_resume_rejects_changed_response_mapping(self):
        collector = self.collector()
        self.close(collector)
        changed = {**runner.MAPPINGS, "binary_11": {"a": 1, "b": -1}}
        with patch.object(runner, "MAPPINGS", changed):
            with self.assertRaises(ValueError):
                self.collector()

    def test_cli_rejects_zero_concurrency_before_constructing_collector(self):
        with patch("sys.argv", ["run_experiment.py", "--concurrency", "0"]):
            with patch.object(runner, "Collector", side_effect=AssertionError("Collector must not start")):
                with self.assertRaises(SystemExit) as exception:
                    runner.main()
        self.assertEqual(exception.exception.code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
