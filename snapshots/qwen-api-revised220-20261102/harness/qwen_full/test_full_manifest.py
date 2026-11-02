"""Offline protocol/queue tests; no simulator, network, or model is required."""
import json
from pathlib import Path
import tempfile
import unittest

from build_full_manifest import ManifestError, build_manifest, main


class FullManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "checkout"
        cls.assets = Path(cls.tmp.name) / "host-assets"
        cls.scripts = cls.root / "scripts/internal"
        cls.tasks = cls.root / "task/RoboDojo/tasks"
        cls.configs = cls.root / "task/RoboDojo/config"
        for path in [cls.scripts, cls.tasks, cls.configs]:
            path.mkdir(parents=True)
        cls.dimensions = {name: tuple(f"{prefix}_{index:02}" for index in range(count))
                          for name, prefix, count in [("generalization", "gen", 12),
                                                      ("precision", "precision", 8),
                                                      ("long-horizon", "long", 8),
                                                      ("memory", "memory", 6), ("open", "open", 8)]}
        (cls.scripts / "task_inventory.py").write_text("DIMENSION_TASKS: dict = " + repr(cls.dimensions))
        summary_dimensions = {key.title(): list(value) for key, value in cls.dimensions.items()}
        (cls.scripts / "summarize_result.py").write_text(
            "DIMENSIONS = " + repr(summary_dimensions) + "\nEXPECTED_SEEDS = [0, 1, 2]\n"
            "STANDALONE_EPISODES = 50\nPAIRED_HALF_EPISODES = 25\n")
        base = {task for tasks in cls.dimensions.values() for task in tasks}
        cls.names = sorted(base | {task + "_random" for task in cls.dimensions["generalization"]})
        cls.options_text = "common:\n  eval_nums: 50\n  robot_config: \"dual_x5\"\n  robot_self_collision: true\ntasks:\n"
        for task in cls.names:
            if task.startswith("gen_"):
                cls.options_text += f"  {task}:\n    eval_nums: 25\n"
        (cls.configs / "_task.yml").write_text(cls.options_text)
        cls.budgets = {}
        for idx, task in enumerate(cls.names):
            cls.budgets[task] = 200 + idx * 10
            (cls.tasks / (task + ".py")).write_text(
                f"class {task}:\n    def __init__(self):\n        self.step_lim = {cls.budgets[task]}\n")
            (cls.configs / (task + ".yml")).write_text("Rigid: []\n")
        # 54*50-10 = 2,690 candidates per seed; 8,070 across all three.
        # Suffix gaps verify IDs follow enumeration, not the filename suffix.
        for seed in (0, 1, 2):
            directory = cls.assets / "Eval_Layout/RoboDojo/arx_x5" / str(seed)
            directory.mkdir(parents=True)
            for task in cls.names:
                count = 40 if task == "gen_00" else 50
                for idx in range(count):
                    (directory / f"{task}_{idx * 2}.json").write_text("{}")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def build(self, **kwargs):
        return build_manifest(self.root, self.assets, **kwargs)

    def test_full_native_counts_and_candidate_authorization(self):
        result = self.build()
        self.assertEqual((result["family_count"], result["task_count"], result["group_count"]), (42, 54, 162))
        self.assertEqual(result["total_target_episodes"], 6300)
        self.assertEqual(result["total_candidate_episodes"], 8070)
        self.assertEqual(len(result["episode_ids"]), 8070)
        self.assertEqual(len(result["episode_seeds"]), 8070)
        for group in result["groups"]:
            self.assertEqual(group["step_limit"], self.budgets[group["task"]])
            self.assertEqual(group["target_episodes"], 25 if group["dimension"] == "generalization" else 50)
            self.assertEqual(len(group["episode_ids"]), group["candidate_count"])
            self.assertEqual(group["layouts"][1]["layout_id"], 1)
            self.assertEqual(group["layouts"][1]["filename_suffix"], 2)
            self.assertEqual(group["episode_ids"][1], group["group_id"] + "_l1")
            self.assertEqual(result["episode_seeds"][group["episode_ids"][-1]], group["eval_seed"])
            self.assertIs(group["native_options"]["robot_self_collision"], True)
        self.assertIsNone(result["source"]["git_head"])

    def test_priority_changes_order_not_scope(self):
        baseline = self.build()
        result = self.build(priority=["open_07_s2", "gen_00_random"])
        self.assertEqual(result["groups"][0]["group_id"], "open_07_s2")
        self.assertEqual([group["group_id"] for group in result["groups"][1:4]],
                         [f"gen_00_random_s{seed}" for seed in (0, 1, 2)])
        self.assertEqual(result["episode_ids"], baseline["episode_ids"])
        with self.assertRaises(ManifestError):
            self.build(priority=["unknown_task"])

    def test_native_count_mismatch_rejected(self):
        path = self.configs / "_task.yml"
        try:
            path.write_text(self.options_text.replace("eval_nums: 50", "eval_nums: 49"))
            with self.assertRaisesRegex(ManifestError, "eval_nums"):
                self.build()
        finally:
            path.write_text(self.options_text)

    def test_missing_candidates_are_not_silently_scored(self):
        path = self.assets / "Eval_Layout/RoboDojo/arx_x5/0/open_00_0.json"
        try:
            path.unlink()
            with self.assertRaisesRegex(ManifestError, "candidates <"):
                self.build()
        finally:
            path.write_text("{}")

    def test_duplicate_numeric_suffix_rejected(self):
        path = self.assets / "Eval_Layout/RoboDojo/arx_x5/0/gen_00_00.json"
        try:
            path.write_text("{}")
            with self.assertRaisesRegex(ManifestError, "duplicate numeric"):
                self.build()
        finally:
            path.unlink()

    def test_dynamic_action_budget_rejected(self):
        path = self.tasks / "open_00.py"
        original = path.read_text()
        try:
            path.write_text(original.replace(str(self.budgets["open_00"]), "int('200')"))
            with self.assertRaisesRegex(ManifestError, "dynamic step_lim"):
                self.build()
        finally:
            path.write_text(original)

    def test_native_budget_from_same_file_common_mixin(self):
        path = self.tasks / "open_00.py"
        original = path.read_text()
        try:
            path.write_text(original.replace("class open_00:", "class OpenCommon:")
                            + "\nclass open_00(OpenCommon):\n    pass\n")
            group = next(group for group in self.build()["groups"] if group["task"] == "open_00")
            self.assertEqual(group["step_limit"], self.budgets["open_00"])
        finally:
            path.write_text(original)

    def test_cli_writes_json(self):
        output = Path(self.tmp.name) / "output/manifest.json"
        main(["--robodojo-root", str(self.root), "--assets-root", str(self.assets), "--output", str(output)])
        self.assertEqual(json.loads(output.read_text())["total_target_episodes"], 6300)


if __name__ == "__main__":
    unittest.main()
