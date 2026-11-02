"""CPU-only recorded-replay and accounting tests; no actual inference or GPUs."""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_model_driver import MODEL, NATIVE_CONTEXT_LENGTH, build_chat_requests, sha
from local_model_driver import redact
from mtp_replay_benchmark import (ReplayHTTP, benchmark, metric_delta, parse_metrics,
                                prepare_case, run_batch, server_command, validate_server_log)


def request_payload(image="data:image/png;base64,AA=="):
    return build_chat_requests([{"role": "system", "content": "recorded system"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image, "detail": "high"}}]}],
        seed=2, remaining_context=NATIVE_CONTEXT_LENGTH - 31570)[0]


def log_text(variant="bf16"):
    args = {"host": "127.0.0.1", "port": 8001, "dtype": "bfloat16", "max_model_len": NATIVE_CONTEXT_LENGTH,
            "tensor_parallel_size": 2, "max_num_seqs": 2}
    spec = "None" if variant == "bf16" else f"SpeculativeConfig(method='mtp', model='/weights', num_spec_tokens={variant[-1]})"
    return (f"INFO non-default args: {args!r}\nINFO Initializing a V1 LLM engine config: "
            f"speculative_config={spec}, tokenizer='/weights', quantization=None\n")


class PrepareTests(unittest.TestCase):
    def test_reconstructs_exact_original_request_with_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = request_payload()
            public = {"public_observation": {"images": [{"camera": "head", "data_url": "data:image/png;base64,AA=="}]}}
            (root / "public_input_0000.json").write_text(json.dumps(public))
            common = {"episode_id": "a_s2_l0", "attempt": 1, "iteration": 6}
            rows = [{**common, "event": "program_input", "request": redact(payload), "request_sha256": sha(payload)},
                    {**common, "event": "input_token_count", "count": 31570, "max_model_len": NATIVE_CONTEXT_LENGTH}]
            (root / "usage.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            summary = prepare_case(root / "usage.jsonl", root, "a_s2_l0", 1, 6, root / "case.json")
            restored = json.loads((root / "case.json").read_text())
            self.assertEqual(restored["request"], payload)
            self.assertEqual(summary["source"]["request_sha256"], sha(payload))
            self.assertEqual(summary["prompt_tokens"], 31570)
            self.assertEqual(summary["image_count"], 1)
            rows[0]["request_sha256"] = "wrong"
            (root / "bad.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                prepare_case(root / "bad.jsonl", root, "a_s2_l0", 1, 6, root / "badcase.json")

    def test_server_command_changes_only_speculation(self):
        base = server_command("/python", "/weights", 8001, "bf16")
        for variant, depth in (("mtp1", 1), ("mtp3", 3)):
            cmd = server_command("/python", "/weights", 8001, variant)
            self.assertEqual(cmd[:-2], base)
            self.assertEqual(json.loads(cmd[-1]), {"method": "mtp", "num_speculative_tokens": depth})

    def test_validates_real_server_variant_and_rejects_production_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "server.log"
            p.write_text(log_text("mtp3"))
            validate_server_log(p, "http://127.0.0.1:8001/v1", "mtp3")
            with self.assertRaises(ValueError):
                validate_server_log(p, "http://127.0.0.1:8001/v1", "bf16")
            with self.assertRaises(ValueError):
                validate_server_log(p, "http://127.0.0.1:8002/v1", "mtp3")
            with self.assertRaisesRegex(ValueError, "production"):
                validate_server_log(Path(tmp)/"full_results/x/allocations/1/server.log", "http://127.0.0.1:8001/v1", "mtp3")


class TransportTests(unittest.TestCase):
    def test_sse_counts_reasoning_ttft_and_never_executes_code(self):
        packets = [{"id": "x", "model": MODEL, "choices": [{"delta": {"reasoning": "first thought"}}]},
                   {"id": "x", "model": MODEL, "choices": [{"delta": {"content": json.dumps({
                       "code": "raise RuntimeError('must not execute')", "summary": "test", "give_up": False})},
                       "finish_reason": "stop"}]},
                   {"id": "x", "model": MODEL, "choices": [], "usage": {"prompt_tokens": 31570, "completion_tokens": 100}}]
        wire = ("".join("data: " + json.dumps(p) + "\n\n" for p in packets) + "data: [DONE]\n\n").encode()
        class Opener:
            def open(self, *args, **kwargs):
                return io.BytesIO(wire)
        http = ReplayHTTP("http://127.0.0.1:8001/v1")
        http.local.opener = Opener()
        result = http.generate(request_payload())
        self.assertEqual(result["output_tokens"], 100)
        self.assertTrue(result["final_json_schema_valid"])
        self.assertFalse(result["generated_code_executed"])
        self.assertGreaterEqual(result["observed_ttft_s"], 0)
        self.assertGreater(result["output_tokens_per_wall_second"], 0)

    def test_acceptance_uses_batch_counter_deltas_and_ignores_created(self):
        before = parse_metrics('vllm:spec_decode_num_draft_tokens_total{engine="0"} 20\n'
                               'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 10\n')
        after = parse_metrics('vllm:spec_decode_num_draft_tokens_total{engine="0"} 120\n'
                              'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 85\n'
                              'vllm:spec_decode_num_draft_tokens_created{engine="0"} 999\n')
        result = metric_delta(before, after)
        self.assertEqual(result["draft_tokens"], 100)
        self.assertEqual(result["accepted_tokens"], 75)
        self.assertEqual(result["acceptance_rate"], .75)


class BenchmarkTests(unittest.TestCase):
    def test_invalid_completed_json_fails_but_timing_cap_truncation_is_expected(self):
        payload = request_payload()
        case = {"source": {"request_sha256": sha(payload)}, "request": payload,
                "verified_prompt_tokens": 31570, "image_count": 1}

        class FakeHTTP:
            def __init__(self, finish):
                self.finish = finish
            def metrics(self):
                return {}
            def generate(self, request):
                return {"output_tokens": request["max_tokens"],
                        "usage": {"prompt_tokens": 31570}, "finish_reason": self.finish,
                        "final_json_schema_valid": False,
                        "benchmark_cap_reached": self.finish == "length"}

        with patch("mtp_replay_benchmark.gpu_snapshot", return_value={"csv": "mock"}):
            completed = run_batch([case], 1, 128, 0, False, FakeHTTP("stop"), ["GPU-test"])
            truncated = run_batch([case], 1, 128, 0, False, FakeHTTP("length"), ["GPU-test"])
        self.assertFalse(completed["requests"][0]["ok"])
        self.assertIn("production JSON schema", completed["requests"][0]["error"])
        self.assertTrue(truncated["requests"][0]["ok"])
        self.assertTrue(truncated["requests"][0]["benchmark_cap_reached"])
        self.assertFalse(truncated["requests"][0]["final_json_schema_valid"])

    def test_concurrent_batches_keep_sampling_and_cap_only_this_test(self):
        class FakeHTTP:
            def __init__(self):
                self.payloads = []
                self.tokens = 0
                self.barrier = threading.Barrier(2)
            def tokenize(self, payload):
                return {"count": 31570, "max_model_len": NATIVE_CONTEXT_LENGTH}
            def metrics(self):
                return {"vllm:generation_tokens_total": self.tokens}
            def generate(self, payload):
                self.payloads.append(payload)
                self.barrier.wait(timeout=3)
                time.sleep(.01)
                self.tokens += 10
                return {"output_tokens": 10, "usage": {"prompt_tokens": 31570},
                        "final_json_schema_valid": True, "benchmark_cap_reached": False,
                        "finish_reason": "stop", "wall_s": .01}
        http = FakeHTTP()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "server.log"; log.write_text(log_text())
            case_files = []
            for index in range(2):
                payload = request_payload(f"data:image/png;base64,A{index}==")
                case = {"source": {"request_sha256": sha(payload), "episode_id": str(index)},
                        "request": payload, "image_count": 1, "prompt_tokens": 31570}
                p = root / f"case{index}.json"; p.write_text(json.dumps(case)); case_files.append(p)
            with patch("mtp_replay_benchmark.gpu_snapshot", return_value={"csv": "mock"}):
                result = benchmark(case_files, "http://127.0.0.1:8001/v1", log, "bf16", root / "output",
                                   2, 128, 1, 1, ["GPU-test"], _http=http)
            self.assertEqual(result["measured_batches"], 1)
            self.assertEqual(result["output_tokens"], 20)  # excludes warmup20
            self.assertEqual(len(http.payloads), 4)
            self.assertTrue(all(p["max_tokens"] == 128 and p["seed"] == 2 for p in http.payloads))
            self.assertEqual(json.loads(case_files[0].read_text())["request"]["max_tokens"], NATIVE_CONTEXT_LENGTH - 31570)
            measured = json.loads((root / "output/batch_00_001.json").read_text())
            self.assertTrue(measured["metrics_generation_accounting_caught_up"])
            self.assertEqual(len(measured["requests"]), 2)


if __name__ == "__main__":
    unittest.main()
