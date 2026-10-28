"""Focused protocol tests; no model, simulator, network, or paid API calls."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from robodata.agents.full_model_driver import (
    FullQwenDriver, InferenceInfrastructureError, LoopbackStreamingHTTP,
    MODEL, ModelEpisodeFailure, NATIVE_CONTEXT_LENGTH, build_chat_requests,
)
from robodata.agents.full_relay import FullRelayClient, serve_full_relay


OBS = {"robot_state": {"q": [0.0]}, "remaining_actions": 200,
       "images": [{"camera": "front", "data_url": "data:image/png;base64,AA=="}]}
PROGRAM = {"code": "result = {'observed': True}", "summary": "observe", "done": False}


class FakeTransport:
    def __init__(self):
        self.prompts, self.generations = [], []
        self.count_override = None
        self.context = NATIVE_CONTEXT_LENGTH
        self.finish = "stop"
        self.content = json.dumps(PROGRAM)
        self.completion = 9001  # Deliberately above the previous 8192 pilot cap.
        self.prompt_offset = 0

    def __call__(self, path, payload, progress=None):
        if path == "/tokenize":
            self.prompts.append(copy.deepcopy(payload))
            self.count = self.count_override if self.count_override is not None else len(payload["messages"]) * 100
            return {"count": self.count, "max_model_len": self.context}
        self.generations.append(copy.deepcopy(payload))
        if progress:
            progress({"chunks_received": 2, "reasoning_delta": "private reasoning", "content_delta": self.content})
        completion = payload["max_tokens"] if self.completion == "remaining" else self.completion
        return {"model": MODEL, "id": f"test-{len(self.generations)}", "usage": {
            "prompt_tokens": self.count + self.prompt_offset, "completion_tokens": completion,
            "total_tokens": self.count + self.prompt_offset + completion},
            "choices": [{"finish_reason": self.finish, "message": {
                "content": self.content, "reasoning": "private reasoning"}}]}


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.transport = FakeTransport()
        self.driver = FullQwenDriver(Path(self.tmp.name) / "usage.jsonl",
            ["stack_s0_l0", "stack_s2_l55"], episode_seeds={"stack_s0_l0": 0, "stack_s2_l55": 2},
            api_documentation="Robot API documentation", _transport=self.transport)

    def tearDown(self):
        self.tmp.cleanup()

    def next(self, episode="stack_s0_l0", feedback=None):
        return self.driver.next_program(episode, "Stack blocks", OBS, feedback)

    def test_more_than_eight_turns_and_8192_output_tokens_preserve_history(self):
        feedback = None
        for index in range(11):
            proposal = self.next(feedback=feedback)
            self.assertEqual(proposal.iteration, index + 1)
            self.assertEqual(proposal.usage["output_tokens"], 9001)
            feedback = {"stdout": "feedback " + str(index), "result": {"observed": True}}
        request = self.transport.generations[-1]
        self.assertEqual(len(request["messages"]), 22)  # system + 10 exchanges + current user
        self.assertEqual(sum(item.get("type") == "image_url" for message in request["messages"]
            if isinstance(message["content"], list) for item in message["content"]), 11)
        self.assertNotIn("iterations_remaining", json.dumps(request))
        self.assertEqual(request["max_tokens"], NATIVE_CONTEXT_LENGTH - 2200)
        self.assertTrue(request["stream"])
        self.assertNotIn("private reasoning", json.dumps(request["messages"][1:]))
        self.assertEqual(proposal.budget["calls"], {"stack_s0_l0": 11})

    def test_actual_policy_seed_and_full_feedback(self):
        self.next("stack_s2_l55")
        self.assertEqual(self.transport.generations[-1]["seed"], 2)
        large_feedback = {"stdout": "x" * 220_000}
        self.next("stack_s2_l55", large_feedback)
        self.assertIn("x" * 220_000, self.transport.generations[-1]["messages"][-1]["content"][0]["text"])

    def test_native_boundary_is_episode_local(self):
        self.transport.finish, self.transport.completion = "length", "remaining"
        with self.assertRaises(ModelEpisodeFailure) as caught:
            self.next()
        self.assertEqual(caught.exception.code, "native_context_exhausted")
        self.assertFalse(self.driver.infrastructure_stopped)
        self.driver.finish_episode("stack_s0_l0", "official_result_committed")
        self.transport.finish, self.transport.completion = "stop", 9001
        self.next("stack_s2_l55")
        self.assertEqual(self.driver.completed_calls, 2)

    def test_an_unexpected_short_server_cap_is_infrastructure(self):
        self.transport.finish, self.transport.completion = "length", 8192
        with self.assertRaises(InferenceInfrastructureError) as caught:
            self.next()
        self.assertEqual(caught.exception.code, "unexpected_generation_cap")
        self.assertTrue(self.driver.infrastructure_stopped)

    def test_full_prompt_exhaustion_does_not_start_generation(self):
        self.transport.count_override = NATIVE_CONTEXT_LENGTH
        with self.assertRaises(ModelEpisodeFailure):
            self.next()
        self.assertEqual(self.transport.generations, [])
        self.assertEqual(self.driver.calls["stack_s0_l0"], 0)

    def test_accounting_mismatch_stops_worker_without_retry(self):
        self.transport.prompt_offset = 1
        with self.assertRaises(InferenceInfrastructureError) as caught:
            self.next()
        self.assertEqual(caught.exception.code, "accounting_mismatch")
        with self.assertRaises(InferenceInfrastructureError):
            self.next("stack_s2_l55")
        self.assertEqual(len(self.transport.generations), 1)

    def test_completed_and_interrupted_attempts_are_distinct(self):
        self.transport.content = json.dumps({"code": "", "summary": "done", "done": True})
        self.next()
        self.assertNotIn("stack_s0_l0", self.driver.history)
        # The native scoring write can fail after the model has ended.
        self.driver.abandon_episode("stack_s0_l0", "infrastructure_interrupted")
        self.next()
        self.assertEqual(self.driver.attempts["stack_s0_l0"], 2)
        self.assertEqual(self.driver.lifetime_calls["stack_s0_l0"], 2)
        self.driver.finish_episode("stack_s0_l0", "official_result_committed")
        self.driver.finish_episode("stack_s0_l0", "official_result_committed")  # idempotent
        with self.assertRaises(InferenceInfrastructureError):
            self.driver.abandon_episode("stack_s0_l0", "native_unstable")
        rows = [json.loads(line) for line in self.driver.path.read_text().splitlines()]
        self.assertEqual(len([row for row in rows if row["event"] == "episode_closed"]), 1)
        self.assertEqual([row["attempt"] for row in rows if row["event"] == "generation_started"], [1, 2])

    def test_malformed_program_ends_one_episode(self):
        self.transport.content = "not JSON"
        with self.assertRaises(ModelEpisodeFailure) as caught:
            self.next()
        self.assertEqual(caught.exception.code, "invalid_model_program")
        self.transport.content = json.dumps(PROGRAM)
        self.next("stack_s2_l55")
        self.assertFalse(self.driver.infrastructure_stopped)

    def test_serving_context_mismatch_is_not_a_model_failure(self):
        self.transport.context = 128000
        with self.assertRaises(InferenceInfrastructureError) as caught:
            self.next()
        self.assertEqual(caught.exception.code, "context_configuration_mismatch")
        self.assertEqual(self.transport.generations, [])


class StreamTests(unittest.TestCase):
    def stream(self, packets, done=True):
        text = "".join("data: " + json.dumps(packet) + "\n\n" for packet in packets)
        return io.BytesIO((text + ("data: [DONE]\n\n" if done else "")).encode())

    def packets(self):
        return [
            {"id": "abc", "model": MODEL, "choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}]},
            {"id": "abc", "model": MODEL, "choices": [{"index": 0, "delta": {"content": json.dumps(PROGRAM)}, "finish_reason": "stop"}]},
            {"id": "abc", "model": MODEL, "choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 9000}},
        ]

    def test_long_thinking_stream_collects_final_answer_and_usage(self):
        progress = []
        result = LoopbackStreamingHTTP.read_sse(self.stream(self.packets()), progress.append)
        self.assertEqual(result["choices"][0]["message"]["reasoning"], "thinking")
        self.assertEqual(json.loads(result["choices"][0]["message"]["content"]), PROGRAM)
        self.assertEqual(result["usage"]["completion_tokens"], 9000)
        self.assertEqual(progress[-1]["chunks_received"], 3)

    def test_partial_stream_never_becomes_a_scored_model_failure(self):
        with self.assertRaises(InferenceInfrastructureError) as caught:
            LoopbackStreamingHTTP.read_sse(self.stream(self.packets(), done=False))
        self.assertEqual(caught.exception.code, "incomplete_stream")

    def test_transport_is_strictly_local_and_credential_free(self):
        for url in ("https://api.openai.com/v1", "http://example.com/v1",
                    "http://user:password@127.0.0.1:8000/v1", "http://127.0.0.1/v1?key=x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                LoopbackStreamingHTTP(url)
        for url in ("http://127.0.0.1:8000/v1", "http://[::1]:8000/v1"):
            LoopbackStreamingHTTP(url)

    def test_normal_server_queueing_has_no_default_network_deadline(self):
        url = "http://127.0.0.1:8000/v1"
        self.assertIsNone(LoopbackStreamingHTTP(url).idle_timeout)
        self.assertIsNone(LoopbackStreamingHTTP(url, socket_idle_timeout_s=None).idle_timeout)
        self.assertEqual(LoopbackStreamingHTTP(url, socket_idle_timeout_s=60).idle_timeout, 60.0)
        for invalid in (0, -1, float("inf"), float("nan"), True, "60"):
            with self.subTest(timeout=invalid), self.assertRaises(ValueError):
                LoopbackStreamingHTTP(url, socket_idle_timeout_s=invalid)

    def test_default_build_has_no_independent_generation_cap(self):
        generation, tokenize = build_chat_requests([], seed=2)
        self.assertNotIn("max_tokens", generation)
        self.assertNotIn("max_completion_tokens", generation)
        self.assertEqual(generation["chat_template_kwargs"], tokenize["chat_template_kwargs"])

    def test_interface_smoke_uses_full_driver_without_robot_execution(self):
        from robodata.agents.full_model_smoke import smoke
        transport = FakeTransport()
        transport.content = json.dumps({"code": "result = {'colors': ['red', 'green', 'blue']}",
                                       "summary": "colors", "done": False})
        with tempfile.TemporaryDirectory() as tmp:
            report = smoke(Path(tmp) / "smoke.json", _transport=transport)
            self.assertTrue(report["ok"])
            self.assertFalse(report["measured_episode"])
            self.assertEqual(report["image_count"], 3)
            self.assertEqual(report["usage"]["output_tokens"], 9001)
            self.assertEqual(transport.generations[0]["max_tokens"], NATIVE_CONTEXT_LENGTH - 200)


class RelayTests(unittest.TestCase):
    def test_real_cli_context_failure_continues_next_episode(self):
        # Exercise the actual __main__ entrypoint in a separate interpreter.
        # In-process tests miss duplicated exception classes when full_relay
        # imports full_model_driver after it was executed as __main__.
        calls = []

        class MockVLLM(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((self.path, payload))
                exhaust = "exhaust native context" in json.dumps(payload["messages"])
                prompt = NATIVE_CONTEXT_LENGTH - 10 if exhaust else 100
                if self.path == "/tokenize":
                    body = json.dumps({"count": prompt, "max_model_len": NATIVE_CONTEXT_LENGTH}).encode()
                    content_type = "application/json"
                else:
                    completed = {"code": "", "summary": "complete", "done": True}
                    packets = [
                        {"id": "mock", "model": MODEL, "choices": [{"index": 0,
                            "delta": {"reasoning_content": "mock reasoning"} if exhaust else
                                     {"content": json.dumps(completed)},
                            "finish_reason": "length" if exhaust else "stop"}]},
                        {"id": "mock", "model": MODEL, "choices": [], "usage": {
                            "prompt_tokens": prompt, "completion_tokens": 10 if exhaust else 16}},
                    ]
                    body = ("".join("data: " + json.dumps(p) + "\n\n" for p in packets) +
                            "data: [DONE]\n\n").encode()
                    content_type = "text/event-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), MockVLLM)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = {"usage_jsonl": str(root / "usage.jsonl"),
                          "episode_ids": ["a_s0_l0", "a_s0_l1"],
                          "episode_seeds": {"a_s0_l0": 0, "a_s0_l1": 0},
                          "api_documentation": "mock API", "socket_idle_timeout_s": None,
                          "base_url": f"http://127.0.0.1:{server.server_port}/v1"}
                (root / "config.json").write_text(json.dumps(config))
                process = subprocess.Popen([sys.executable,
                    "-m", "robodata.agents.full_model_driver",
                    "--config", str(root / "config.json"), "--relay-dir", str(root / "relay")],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    deadline = time.monotonic() + 10
                    while not (root / "relay" / "ready.json").exists():
                        if process.poll() is not None or time.monotonic() > deadline:
                            self.fail("CLI relay did not become ready")
                        time.sleep(.02)
                    client = FullRelayClient(root / "relay")
                    with self.assertRaises(ModelEpisodeFailure) as caught:
                        client.next_program("a_s0_l0", "exhaust native context", OBS)
                    self.assertEqual(caught.exception.code, "native_context_exhausted")
                    self.assertIsNone(process.poll())
                    self.assertFalse(client.stopped)
                    client.finish_episode("a_s0_l0", "official_result_committed")
                    proposal = client.next_program("a_s0_l1", "ordinary task", OBS)
                    self.assertTrue(proposal.done)
                    client.finish_episode("a_s0_l1", "official_result_committed")
                    client.stop()
                    stdout, stderr = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 0, stderr)
                    rows = [json.loads(line) for line in (root / "usage.jsonl").read_text().splitlines()]
                    self.assertEqual([row["episode_id"] for row in rows
                                      if row["event"] == "model_episode_failed"], ["a_s0_l0"])
                    self.assertFalse(any(row["event"] == "infrastructure_stopped" for row in rows))
                    self.assertEqual(len([row for row in rows if row["event"] == "episode_closed"]), 2)
                    generations = [payload for path, payload in calls if path == "/v1/chat/completions"]
                    self.assertEqual(len(generations), 2)
                    self.assertEqual(generations[0]["max_tokens"], 10)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=10)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(3)

    def test_infrastructure_error_stops_relay_without_scoring_or_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            transport = FakeTransport()
            transport.prompt_offset = 1
            driver = FullQwenDriver(Path(tmp) / "usage.jsonl", ["a_s0_l0"],
                episode_seeds={"a_s0_l0": 0}, api_documentation="API", _transport=transport)
            errors = []
            def serve():
                try:
                    serve_full_relay(driver, Path(tmp) / "relay")
                except InferenceInfrastructureError as exc:
                    errors.append(exc.code)
            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            client = FullRelayClient(Path(tmp) / "relay")
            with self.assertRaises(InferenceInfrastructureError) as caught:
                client.next_program("a_s0_l0", "task", OBS)
            self.assertEqual(caught.exception.code, "accounting_mismatch")
            self.assertTrue(client.stopped)
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, ["accounting_mismatch"])
            self.assertEqual(driver.failed, {})
            self.assertEqual(driver.finished, set())
            self.assertEqual(len(transport.generations), 1)

    def test_model_failure_does_not_stop_next_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            transport = FakeTransport()
            transport.content = "invalid JSON"
            driver = FullQwenDriver(Path(tmp) / "usage.jsonl", ["a_s0_l0", "a_s0_l1"],
                episode_seeds={"a_s0_l0": 0, "a_s0_l1": 0}, api_documentation="API", _transport=transport)
            errors = []
            def serve():
                try:
                    serve_full_relay(driver, Path(tmp) / "relay")
                except BaseException as exc:
                    errors.append(exc)
            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            client = FullRelayClient(Path(tmp) / "relay")
            self.assertIsNone(client.wait_ready()["absolute_request_timeout_s"])
            try:
                with self.assertRaises(ModelEpisodeFailure):
                    client.next_program("a_s0_l0", "task", OBS)
                self.assertFalse(client.stopped)
                client.finish_episode("a_s0_l0", "official_result_committed")
                transport.content = json.dumps(PROGRAM)
                self.assertEqual(client.next_program("a_s0_l1", "task", OBS).iteration, 1)
                client.abandon_episode("a_s0_l1", "native_unstable")
                self.assertEqual(client.next_program("a_s0_l1", "task", OBS).iteration, 1)
                client.finish_episode("a_s0_l1", "official_result_committed")
            finally:
                if thread.is_alive() and not client.stopped:
                    client.stop()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(list((Path(tmp) / "relay" / "requests").glob("*.json")), [])
            self.assertEqual(len(list((Path(tmp) / "relay" / "processed_requests").glob("*.json"))), 7)


if __name__ == "__main__":
    unittest.main()
