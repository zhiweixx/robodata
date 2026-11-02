"""CPU-only protocol checks: fake transports only, no model/network calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from local_model_driver import (DriverStopped, EPISODES, LocalQwenDriver,
                                LoopbackHTTP, MODEL, THINKING, chat_content)
from model_driver import BudgetedAstraDriver, OUTPUT_FORMAT
from test_model_driver import FakeSDK


OBS = {"robot_state": {"joint_positions": {"right": [0.] * 6}},
       "camera_calibration": {"cam_head": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
       "images": [{"camera": name, "data_url": "data:image/png;base64,AAAA"}
                  for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")],
       "elapsed_sim_s": 0., "remaining_actions": 200}


class FakeLocal:
    def __init__(self):
        self.requests = []
        self.prompt_count = 1000
        self.actual_prompt_count = None
        self.max_model_len = 136192
        self.completion_count = 500
        self.finish = "stop"
        self.fail = False
        self.output = json.dumps({"code": "result = get_robot_state()", "summary": "Inspect robot", "give_up": False})

    def __call__(self, path, payload):
        self.requests.append((path, copy.deepcopy(payload)))
        if path == "/tokenize":
            return {"count": self.prompt_count, "max_model_len": self.max_model_len}
        if self.fail:
            raise TimeoutError("offline simulated inference timeout")
        return {"id": "local_mock", "model": MODEL,
                "choices": [{"finish_reason": self.finish, "message": {
                    "role": "assistant", "content": self.output,
                    "reasoning_content": "offline reasoning fixture; never reuse as context"}}],
                "usage": {"prompt_tokens": self.prompt_count if self.actual_prompt_count is None else self.actual_prompt_count,
                          "completion_tokens": self.completion_count}}

    @property
    def generations(self):
        return [payload for path, payload in self.requests if path == "/v1/chat/completions"]


class LocalDriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "local.jsonl"
        self.fake = FakeLocal()

    def driver(self, **options):
        return LocalQwenDriver(self.path, api_documentation="get_robot_state() -> public state",
                               _transport=self.fake, **options)

    def step(self, driver, feedback=None, episode=EPISODES[0]):
        return driver.next_program(episode, "Lift the cup", OBS, previous_result=feedback)

    def test_same_prompt_contents_as_original_astra_fixture(self):
        local = self.driver()
        sdk = FakeSDK()
        reference = BudgetedAstraDriver(Path(self.tmp.name) / "reference.jsonl", EPISODES,
                                       "get_robot_state() -> public state", max_iterations=8,
                                       _sdk_factory=lambda key: sdk)
        with patch.object(reference._module, "_read_key", return_value="offline-fixture"):
            for feedback in (None, {"program_ok": True, "result": {"seen": True}}):
                self.step(local, feedback)
                reference.next_program(EPISODES[0], "Lift the cup", OBS, previous_result=feedback)
        for qwen, astra in zip(self.fake.generations, sdk.creates):
            expected = [{"role": "system", "content": astra["instructions"]}]
            expected += [{**message, "content": chat_content(message["content"])
                          if message["role"] == "user" else message["content"]}
                         for message in astra["input"]]
            self.assertEqual(qwen["messages"], expected)
            self.assertEqual(qwen["response_format"]["json_schema"]["schema"], OUTPUT_FORMAT["format"]["schema"])
            self.assertNotIn("tools", qwen)
            self.assertEqual(qwen["max_tokens"], 8192)
            self.assertEqual(qwen["chat_template_kwargs"], THINKING)
            self.assertEqual(qwen["reasoning_effort"], "medium")

    def test_all_eight_turns_retain_24_images_but_no_reasoning_history(self):
        driver = self.driver()
        for i in range(8):
            proposal = self.step(driver, None if i == 0 else {"program_ok": True})
        messages = self.fake.generations[-1]["messages"]
        images = [part for message in messages if isinstance(message["content"], list)
                  for part in message["content"] if part["type"] == "image_url"]
        self.assertEqual(len(images), 24)
        self.assertNotIn("reasoning_content", json.dumps(messages))
        self.assertEqual(proposal.budget["api_cost_usd"], 0)
        with self.assertRaises(DriverStopped):
            self.step(driver, {"program_ok": True})
        self.assertEqual(len(self.fake.generations), 8)

    def test_no_generation_if_exact_input_cap_exceeded(self):
        self.fake.prompt_count = 128001
        with self.assertRaises(DriverStopped):
            self.step(self.driver())
        self.assertFalse(self.fake.generations)

    def test_full_input_and_output_caps_are_accepted(self):
        self.fake.prompt_count = 128000
        self.fake.completion_count = 8192
        proposal = self.step(self.driver())
        self.assertEqual(proposal.usage["input_tokens"], 128000)
        self.assertEqual(proposal.usage["output_tokens"], 8192)
        self.assertFalse(proposal.usage["token_count_approximate"])

    def test_insufficient_server_context_fails_before_generation(self):
        self.fake.max_model_len = 8192
        with self.assertRaises(DriverStopped):
            self.step(self.driver())
        self.assertFalse(self.fake.generations)

    def test_tokenize_usage_disagreement_never_returns_program(self):
        self.fake.actual_prompt_count = 1001
        with self.assertRaises(DriverStopped):
            self.step(self.driver())
        self.assertEqual(len(self.fake.generations), 1)
        self.assertIn('"preflight_matches_generation_usage":false', self.path.read_text())

    def test_truncation_or_bad_json_never_returns_program(self):
        for index, (finish, output) in enumerate((("length", self.fake.output), ("stop", "<think>unfinished"))):
            self.path = Path(self.tmp.name) / f"bad{index}.jsonl"
            self.fake = FakeLocal()
            self.fake.finish, self.fake.output = finish, output
            driver = self.driver()
            with self.assertRaises(DriverStopped):
                self.step(driver)
            with self.assertRaises(DriverStopped):
                self.step(driver)
            self.assertEqual(len(self.fake.generations), 1)

    def test_transport_failure_no_retry_or_restart(self):
        self.fake.fail = True
        driver = self.driver()
        with self.assertRaises(TimeoutError):
            self.step(driver)
        with self.assertRaises(DriverStopped):
            self.step(driver)
        with self.assertRaises(FileExistsError):
            self.driver()
        self.assertEqual(len(self.fake.generations), 1)

    def test_privileged_input_rejected_before_any_request(self):
        with self.assertRaises(ValueError):
            self.driver().next_program(EPISODES[0], "Lift the cup", {"robot_state": {"reward": 1}})
        self.assertFalse(self.fake.requests)

    def test_audit_keeps_response_but_hashes_inline_images(self):
        proposal = self.step(self.driver())
        records = list(map(json.loads, self.path.read_text().splitlines()))
        self.assertEqual([r["event"] for r in records], ["suite_initialized", "program_input", "input_token_count",
                                                          "generation_started", "raw_response", "generation_usage"])
        text = self.path.read_text()
        self.assertNotIn("data:image/png;base64,AAAA", text)
        self.assertIn("sha256_data_url", text)
        self.assertIn("offline reasoning fixture", text)
        self.assertTrue(proposal.usage["preflight_matches_generation_usage"])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_external_or_credentialed_urls_forbidden(self):
        for url in ("https://api.openai.com/v1", "http://example.com/v1", "http://secret@localhost:8000/v1",
                    "http://localhost:8000/v1?api_key=secret"):
            with self.assertRaises(ValueError):
                LoopbackHTTP(url, 30)

    def test_tokenization_and_generation_have_separate_bounded_timeouts(self):
        http = LoopbackHTTP("http://127.0.0.1:8000/v1", 400, 60)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"count": 5}'
        with patch.object(http.opener, "open", return_value=response) as opened:
            http("/tokenize", {})
            self.assertEqual(opened.call_args.kwargs["timeout"], 60)
            http("/v1/chat/completions", {})
            self.assertEqual(opened.call_args.kwargs["timeout"], 400)
        for generation, tokenize in ((401, 60), (400, 61), (0, 60), (400, float("nan"))):
            with self.assertRaises(ValueError):
                LoopbackHTTP("http://127.0.0.1:8000/v1", generation, tokenize)

    def test_count_and_generation_template_flags_match_pinned_server(self):
        self.step(self.driver())
        tokenize, generation = [payload for _, payload in self.fake.requests]
        for key in ("messages", "chat_template_kwargs", "add_generation_prompt",
                    "continue_final_message", "add_special_tokens"):
            self.assertEqual(tokenize[key], generation[key])
        self.assertNotIn("tools", tokenize)
        self.assertNotIn("tools", generation)
        config = json.loads(self.path.read_text().splitlines()[0])["config"]
        self.assertEqual(config["tokenize_timeout_s"], 60)
        self.assertEqual(config["request_timeout_s"], 400)

    def test_isolated_three_image_smoke_does_not_create_pilot_ledger(self):
        from qwen_model_smoke import smoke
        self.fake.output = json.dumps({"code": 'result = {"colors": ["red", "green", "blue"]}',
                                       "summary": "Colors identified", "give_up": False})
        output = Path(self.tmp.name) / "smoke.json"
        result = smoke(output, _transport=self.fake)
        self.assertTrue(result["ok"])
        self.assertFalse(result["measured_episode"])
        self.assertFalse(self.path.exists())
        self.assertEqual(len(self.fake.generations), 1)
        self.assertEqual(self.fake.generations[0]["chat_template_kwargs"], THINKING)
        self.assertEqual(self.fake.generations[0]["max_tokens"], 8192)
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
