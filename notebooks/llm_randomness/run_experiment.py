"""Collect fresh LLM calls; resume from append-only local audit records."""

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import tomllib
from datetime import datetime, timezone

import httpx


ROOT = Path(__file__).resolve().parent
PROMPTS = [
    "Pick between -1 and 1. Respond only with the number.",
    "Pick between 1 and -1. Respond only with the number.",
    "Sample the number 1 with probability 0.50 and the number -1 with probability 0.50. Respond only with the number and nothing else:",
    "Sample the number -1 with probability 0.50 and the number 1 with probability 0.50. Respond only with the number and nothing else:",
    "Flip a fair coin. Return -1 for heads and 1 for tails. Respond only with -1 or 1.",
    "Flip a fair coin. Return 1 for heads and -1 for tails. Respond only with 1 or -1.",
    "Choose one sign at random with equal probability: negative or positive. Write it as -1 or 1. Output only the number.",
    "Choose one sign at random with equal probability: positive or negative. Write it as 1 or -1. Output only the number.",
    "Flip a fair coin. Respond only with heads or tails.",
    "Flip a fair coin. Respond only with tails or heads.",
    "Pick one at random: A or B. Respond with only the letter.",
    "Pick one at random: B or A. Respond with only the letter.",
    "Generate one random bit. Respond only with 0 or 1.",
    "Mary and Ann played in the park. ___ jumped in the air. Fill in the blank. Respond only with Mary or Ann.",
    "Ann and Mary played in the park. ___ jumped in the air. Fill in the blank. Respond only with Ann or Mary.",
    "Choose randomly between red and blue. Respond with only the color.",
]
MAPPINGS = {f"binary_{i}": {"-1": -1, "1": 1} for i in range(8)}
MAPPINGS.update({
    "binary_8": {"heads": -1, "tails": 1},
    "binary_9": {"heads": -1, "tails": 1},
    "binary_10": {"a": -1, "b": 1},
    "binary_11": {"a": -1, "b": 1},
    "binary_12": {"0": -1, "1": 1},
    "binary_13": {"mary": -1, "ann": 1},
    "binary_14": {"mary": -1, "ann": 1},
    "binary_15": {"red": -1, "blue": 1},
})
INTEGER_PROMPT = "Choose an integer at random from 0 through 999999999. Respond with digits only, no explanation."


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path):
    if not path.exists():
        return []
    result = []
    with path.open() as f:
        for line in f:
            result.append(json.loads(line))
    return result


def parse_response(raw, integer=False):
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if integer:
        return int(raw) if re.fullmatch(r"[0-9]{1,9}", raw) else None
    return int(raw) if raw in ("1", "-1") else None


def spec(phase, prompt_id, prompt, index, size=0):
    return {
        "request_id": f"{phase}:{prompt_id}:{index:08d}",
        "phase": phase, "prompt_id": prompt_id, "prompt": prompt,
        "group_id": index // size if size else None,
        "position": index % size if size else index,
        "planned_index": index,
    }


class Collector:
    def __init__(self, args):
        self.args = args
        self.calibration_calls = getattr(args, "calibration_calls", 1000)
        self.flips_per_sample = getattr(args, "flips_per_sample", 40)
        self.requested_samples = getattr(args, "samples", 2000)
        self.directory = args.data_dir
        self.directory.mkdir(parents=True, exist_ok=True)
        config = tomllib.loads(Path(args.config).expanduser().read_text())
        provider = config["model_providers"][config["model_provider"]]
        self.base_url = provider["base_url"].rstrip("/")
        self.key = os.environ[provider["env_key"]]
        self.path = self.directory / "manifest.json"
        self.responses = {r["request_id"]: r for r in read_jsonl(self.directory / "responses.jsonl")}
        self.attempt_counts = {}
        previous_attempts = read_jsonl(self.directory / "attempts.jsonl")
        for r in previous_attempts + read_jsonl(self.directory / "requests_started.jsonl"):
            self.attempt_counts[r["request_id"]] = max(self.attempt_counts.get(r["request_id"], 0), r["attempt"])
        settings = {
            "model": args.model, "temperature": 1.0, "top_p": 1.0,
            "max_tokens": 32, "n": 1, "stream": False,
        }
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text())
            if self.manifest["settings"] != settings:
                raise ValueError("Cannot resume with changed model or sampling settings")
            if self.manifest["endpoint"] != self.base_url + "/chat/completions":
                raise ValueError("Cannot resume with changed provider endpoint")
            for key, value in [("calibration_calls", self.calibration_calls),
                               ("flips_per_sample", self.flips_per_sample),
                               ("requested_samples_per_method", self.requested_samples)]:
                if self.manifest[key] != value:
                    raise ValueError(f"Cannot resume with changed {key}")
        else:
            self.manifest = {
                "status": "planned", "created_at_utc": utc(),
                "backend": "LiteLLM OpenAI-compatible chat completions",
                "endpoint": self.base_url + "/chat/completions",
                "settings": settings, "seed": None, "system_prompt": None,
                "history": "One user message per call; no prior conversation",
                "independence": "Assumption, not established by fresh calls or diagnostics",
                "source": "Every random input comes from an LLM response; no local random draws",
                "target_mean": 10, "target_standard_deviation": 2,
                "requested_samples_per_method": self.requested_samples,
                "flips_per_sample": self.flips_per_sample,
                "calibration_calls": self.calibration_calls, "pilot_calls_per_prompt": 100,
                "prompts": {f"binary_{i}": p for i, p in enumerate(PROMPTS)},
                "integer_prompt": {"prompt_id": "integer_0", "prompt": INTEGER_PROMPT},
                "selection_rule": "All 100 pilot responses valid, then minimum abs(p-.5); tie by candidate index",
                "grouping": "Immutable planned index and position, never completion order",
                "exclusions": "Whole invalid CLT groups and whole invalid/tied permutation groups; all raw calls retained",
                "phases": {}, "concurrency_history": [],
            }
        selection_path = getattr(args, "selection_from", None)
        if selection_path:
            selection_path = Path(selection_path).resolve()
            source_bytes = selection_path.read_bytes()
            source = json.loads(source_bytes)
            if source["settings"] != settings:
                raise ValueError("Inherited selection used different model or sampling settings")
            if source["endpoint"] != self.base_url + "/chat/completions":
                raise ValueError("Inherited selection used different endpoint")
            selection = source["selection"]
            prompt_id = selection["prompt_id"]
            if source["response_mappings"][prompt_id] != MAPPINGS[prompt_id]:
                raise ValueError("Inherited selection used different response mapping")
            if self.manifest.get("selection", selection) != selection:
                raise ValueError("Cannot change frozen selection on resume")
            provenance = {"source_manifest": str(selection_path),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "description": "Prompt selected in previous experiment; no prior random draws reused"}
            if self.manifest.get("selection_source", provenance) != provenance:
                raise ValueError("Cannot change selection source on resume")
            self.manifest["selection"] = selection
            self.manifest["selection_source"] = provenance
            self.manifest["required_phases"] = ["calibration", "clt"]
        self.manifest["concurrency_history"].append({"at_utc": utc(), "concurrency": args.concurrency})
        for prompt_id, mapping in self.manifest.get("response_mappings", {}).items():
            if MAPPINGS.get(prompt_id) != mapping:
                raise ValueError("Cannot resume with changed response mapping")
        current_prompts = {f"binary_{i}": p for i, p in enumerate(PROMPTS)}
        for prompt_id, prompt in self.manifest.get("prompts", {}).items():
            if current_prompts.get(prompt_id) != prompt:
                raise ValueError("Cannot resume with changed prompt text")
        self.manifest["prompts"] = {f"binary_{i}": p for i, p in enumerate(PROMPTS)}
        self.manifest["response_mappings"] = MAPPINGS
        if "selection_source" not in self.manifest:
            self.manifest["pilot_expansion"] = "First8 literal sign prompts all p>=.99; added8 prespecified binary-label prompts, 100 calls each, before calibration"
        self.manifest.pop("file_sha256", None)
        self.responses_file = (self.directory / "responses.jsonl").open("a", buffering=1)
        self.attempts_file = (self.directory / "attempts.jsonl").open("a", buffering=1)
        self.started_file = (self.directory / "requests_started.jsonl").open("a", buffering=1)
        # A completed provider reply may be journaled just before a process crash.
        # Recover that original reply instead of issuing a second random draw.
        for entry in previous_attempts:
            if entry.get("status") in ("ok", "invalid") and entry["request_id"] not in self.responses:
                recovered = {**entry, "recovered_from_attempt_journal": True}
                self.responses_file.write(json.dumps(recovered) + "\n")
                self.responses[entry["request_id"]] = recovered
        self.client = None
        self.last_progress = time.monotonic()
        self.progress_started = self.last_progress
        self.initial_count = len(self.responses)
        self.stop_reason = None
        self.write_manifest()

    def write_manifest(self):
        self.manifest["updated_at_utc"] = utc()
        self.manifest["terminal_calls"] = len(self.responses)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.manifest, indent=2) + "\n")
        temp.replace(self.path)

    async def call(self, item):
        rid = item["request_id"]
        if rid in self.responses:
            if any(self.responses[rid].get(k) != item.get(k) for k in ("prompt", "phase", "group_id", "position")):
                raise ValueError("Cannot reuse request ID with different prompt or grouping")
            return
        integer = item["phase"] in ("integer_pilot", "permutation")
        body = dict(self.manifest["settings"], messages=[{"role": "user", "content": item["prompt"]}])
        final = None
        for retry in range(5):
            attempt = self.attempt_counts.get(rid, 0) + 1
            self.attempt_counts[rid] = attempt
            start = utc()
            timer = time.monotonic()
            entry = {**item, "attempt": attempt, "started_at_utc": start}
            self.started_file.write(json.dumps(entry) + "\n")
            try:
                response = await self.client.post(
                    self.base_url + "/chat/completions", json=body,
                    headers={"Authorization": "Bearer " + self.key, "Cache-Control": "no-cache"},
                )
                entry["http_status"] = response.status_code
                if response.status_code == 200:
                    obj = response.json()
                    choice = obj.get("choices", [{}])[0]
                    raw = choice.get("message", {}).get("content")
                    value = parse_response(raw, True) if integer else self.manifest["response_mappings"][item["prompt_id"]].get(raw.strip().lower()) if isinstance(raw, str) else None
                    entry.update({
                        "status": "ok" if value is not None and choice.get("finish_reason") == "stop" else "invalid",
                        "raw_response": raw, "value": value,
                        "valid": value is not None and choice.get("finish_reason") == "stop",
                        "provider_response_id": obj.get("id"), "model": obj.get("model"),
                        "system_fingerprint": obj.get("system_fingerprint"),
                        "finish_reason": choice.get("finish_reason"), "usage": obj.get("usage"),
                        "provider_created": obj.get("created"),
                    })
                    final = entry
                else:
                    entry.update({"status": "http_error", "error": response.text.replace(self.key, "[REDACTED]")[:2000]})
                    if response.status_code not in (408, 429, 500, 502, 503, 504):
                        self.stop_reason = f"Non-retryable HTTP {response.status_code}"
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                entry.update({"status": "transport_error", "error_type": type(exc).__name__, "error": str(exc).replace(self.key, "[REDACTED]")[:1000]})
            entry["finished_at_utc"] = utc()
            entry["elapsed_seconds"] = time.monotonic() - timer
            self.attempts_file.write(json.dumps(entry) + "\n")
            if final is not None:
                break
            if self.stop_reason:
                break
            if retry < 4:
                await asyncio.sleep(min(30, 2 ** retry + 0.05 * item["position"]))
        if final is None:
            self.stop_reason = self.stop_reason or "Repeated request failure; resume after resolving service issue"
            return
        self.responses_file.write(json.dumps(final) + "\n")
        self.responses[rid] = final
        if time.monotonic() - self.last_progress >= 15:
            elapsed = time.monotonic() - self.progress_started
            print(json.dumps({"phase": item["phase"], "terminal_calls": len(self.responses),
                "new_calls_per_second": round((len(self.responses) - self.initial_count) / elapsed, 2),
                "utc": utc()}), flush=True)
            self.last_progress = time.monotonic()
            self.write_manifest()

    async def collect(self, specs):
        for s in specs:
            if s["request_id"] in self.responses:
                existing = self.responses[s["request_id"]]
                if any(existing.get(k) != s.get(k) for k in ("prompt", "phase", "group_id", "position")):
                    raise ValueError("Cannot reuse request ID with different prompt or grouping")
        pending = [s for s in specs if s["request_id"] not in self.responses]
        iterator = iter(pending)
        async def worker():
            for item in iterator:
                if self.stop_reason:
                    return
                await self.call(item)
        await asyncio.gather(*(worker() for _ in range(self.args.concurrency)))
        if self.stop_reason:
            raise RuntimeError(self.stop_reason)

    def phase_records(self, phase, prompt_id=None):
        return [r for r in self.responses.values() if r["phase"] == phase and (prompt_id is None or r["prompt_id"] == prompt_id)]

    def finish_phase(self, phase, **extra):
        records = self.phase_records(phase)
        self.manifest["phases"][phase] = {"status": "complete", "calls": len(records),
            "valid_calls": sum(r["valid"] for r in records), **extra}
        self.write_manifest()

    async def pilot(self):
        specs = [spec("binary_pilot", f"binary_{p}", prompt, i) for i in range(100) for p, prompt in enumerate(PROMPTS)]
        await self.collect(specs)
        results = []
        for i, prompt in enumerate(PROMPTS):
            records = self.phase_records("binary_pilot", f"binary_{i}")
            valid = [r for r in records if r["valid"]]
            plus = sum(r["value"] == 1 for r in valid)
            results.append({"prompt_id": f"binary_{i}", "prompt": prompt, "pilot_n": len(records),
                "pilot_valid": len(valid), "pilot_plus": plus, "pilot_p": plus / len(valid) if valid else None})
        eligible = [r for r in results if r["pilot_valid"] == 100]
        if not eligible:
            raise RuntimeError("No pilot prompt had 100 valid replies; inspect pilot before changing plan")
        winner = min(eligible, key=lambda r: abs(r["pilot_p"] - .5))
        self.manifest["pilot_results"] = results
        self.manifest["selection"] = winner
        self.finish_phase("binary_pilot")
        print(json.dumps({"pilot_results": results, "selection": winner}), flush=True)

    async def calibration(self):
        winner = self.manifest["selection"]
        await self.collect([spec("calibration", winner["prompt_id"], winner["prompt"], i) for i in range(self.calibration_calls)])
        records = self.phase_records("calibration")
        if len(records) != self.calibration_calls or not all(r["valid"] for r in records):
            raise RuntimeError(f"Calibration must contain exactly {self.calibration_calls} valid calls; inspect invalid responses")
        plus = sum(r["value"] == 1 for r in records)
        p = plus / self.calibration_calls
        if not 0 < p < 1:
            raise RuntimeError("Calibration has zero variance")
        self.manifest["calibration"] = {"p_hat": p, "mu": 2*p-1,
            "sigma": math.sqrt(4*p*(1-p)), "valid_count": self.calibration_calls, "plus_count": plus,
            "target_mean_calibration_standard_error_approx": 2*math.sqrt(self.flips_per_sample/self.calibration_calls)}
        self.finish_phase("calibration")
        print(json.dumps({"calibration": self.manifest["calibration"]}), flush=True)

    def groups(self, phase, size, distinct):
        groups = {}
        for r in self.phase_records(phase):
            groups.setdefault(r["group_id"], []).append(r)
        accepted = []
        for group_id, rows in groups.items():
            if len(rows) == size and len({r["position"] for r in rows}) == size and all(r["valid"] for r in rows):
                if not distinct or len({r["value"] for r in rows}) == size:
                    accepted.append(group_id)
        return sorted(accepted), groups

    async def integer_pilot(self):
        prompt = self.manifest["integer_prompt"]
        await self.collect([spec("integer_pilot", prompt["prompt_id"], prompt["prompt"], i, 10) for i in range(1000)])
        accepted, groups = self.groups("integer_pilot", 10, True)
        self.finish_phase("integer_pilot", accepted_groups=len(accepted), total_groups=len(groups), acceptance_rate=len(accepted)/len(groups))
        print(json.dumps({"integer_pilot": self.manifest["phases"]["integer_pilot"]}), flush=True)
        if len(accepted) < 10:
            raise RuntimeError("Fewer than 10 percent of integer pilot groups accepted; inspect before large run")

    async def production(self, phase):
        size = self.flips_per_sample if phase == "clt" else 10
        distinct = phase == "permutation"
        prompt = self.manifest["selection"] if phase == "clt" else self.manifest["integer_prompt"]
        accepted, groups = self.groups(phase, size, distinct)
        # Finish any previously started groups before scheduling later group indices.
        if groups:
            await self.collect([spec(phase, prompt["prompt_id"], prompt["prompt"], g*size+i, size)
                for g in sorted(groups) for i in range(size)])
            accepted, groups = self.groups(phase, size, distinct)
        while len(accepted) < self.requested_samples:
            first = max(groups, default=-1) + 1
            # A continuous worker queue avoids waiting for a slow request after
            # every 100 groups. No batch can overshoot the remaining sample quota.
            count = self.requested_samples-len(accepted)
            if first >= 50000:
                raise RuntimeError("Production group safety limit reached")
            await self.collect([spec(phase, prompt["prompt_id"], prompt["prompt"], g*size+i, size)
                for g in range(first, first+count) for i in range(size)])
            accepted, groups = self.groups(phase, size, distinct)
            self.manifest["phases"][phase] = {"status": "running", "calls": len(self.phase_records(phase)),
                "accepted_groups": len(accepted), "total_groups": len(groups)}
            self.write_manifest()
        self.finish_phase(phase, accepted_groups=len(accepted), total_groups=len(groups), selected_group_ids=accepted[:self.requested_samples])

    async def run(self):
        self.manifest["status"] = "running"
        self.write_manifest()
        limits = httpx.Limits(max_connections=self.args.concurrency, max_keepalive_connections=self.args.concurrency)
        async with httpx.AsyncClient(timeout=httpx.Timeout(60, connect=15), limits=limits) as client:
            self.client = client
            if self.args.phase in ("pilot", "all"):
                await self.pilot()
            if self.args.phase in ("calibration", "clt_only", "all"):
                await self.calibration()
            if self.args.phase in ("integer_pilot", "all"):
                await self.integer_pilot()
            if self.args.phase in ("clt", "clt_only", "all"):
                await self.production("clt")
            if self.args.phase in ("permutation", "all"):
                await self.production("permutation")
        required = self.manifest.get("required_phases", ["binary_pilot", "calibration", "integer_pilot", "clt", "permutation"])
        self.manifest["status"] = "complete" if all(self.manifest["phases"].get(p, {}).get("status") == "complete" for p in required) else "partial"
        self.responses_file.flush()
        self.attempts_file.flush()
        self.started_file.flush()
        self.manifest["file_sha256"] = {f: hashlib.sha256((self.directory/f).read_bytes()).hexdigest() for f in ("responses.jsonl", "attempts.jsonl", "requests_started.jsonl")}
        self.write_manifest()
        print(json.dumps({"status": self.manifest["status"], "terminal_calls": len(self.responses)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"new_run")
    parser.add_argument("--config", default="provider.toml")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--calibration-calls", type=int, default=1000)
    parser.add_argument("--flips-per-sample", type=int, default=40)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--selection-from", type=Path)
    parser.add_argument("--phase", choices=["pilot", "calibration", "integer_pilot", "clt", "clt_only", "permutation", "all"], default="all")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if min(args.calibration_calls, args.flips_per_sample, args.samples) < 1:
        parser.error("calibration calls, flips per sample and samples must be positive")
    collector = Collector(args)
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
