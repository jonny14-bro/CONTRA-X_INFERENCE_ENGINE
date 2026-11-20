import importlib.util
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "run.py"
SPEC = importlib.util.spec_from_file_location("pwea_run", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestPWEARunner(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.config_path = self.base / "config.json"
        self.input_path = self.base / "session.jsonl"
        self.input_path.write_text("{}\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def config(self):
        return {
            "schema_version": 1,
            "model": {
                "model_id": "test-model",
                "n_layer": 2,
                "n_expert": 10,
            },
            "inputs": ["session.jsonl"],
            "profile": {
                "output": "out/profile.json",
                "min_prompts": 20,
                "min_workloads": 3,
            },
            "evaluation": {"enabled": False},
        }

    def write_config(self, config):
        self.config_path.write_text(
            json.dumps(config), encoding="utf-8"
        )

    def test_valid_config_resolves_paths_relative_to_config(self):
        self.write_config(self.config())
        result = MODULE.load_config(self.config_path)

        self.assertEqual(result["inputs"], [self.input_path.resolve()])
        self.assertEqual(
            result["profile_output"],
            (self.base / "out/profile.json").resolve(),
        )
        self.assertFalse(result["evaluation_enabled"])

    def test_invalid_json_is_rejected(self):
        self.config_path.write_text("{invalid", encoding="utf-8")

        with self.assertRaises(MODULE.ConfigError):
            MODULE.load_config(self.config_path)

    def test_unsupported_schema_is_rejected(self):
        config = self.config()
        config["schema_version"] = 2
        self.write_config(config)

        with self.assertRaises(MODULE.ConfigError):
            MODULE.load_config(self.config_path)

    def test_duplicate_inputs_are_rejected(self):
        config = self.config()
        config["inputs"] = ["session.jsonl", "./session.jsonl"]
        self.write_config(config)

        with self.assertRaisesRegex(MODULE.ConfigError, "Duplicate"):
            MODULE.load_config(self.config_path)

    def test_missing_input_is_rejected(self):
        config = self.config()
        config["inputs"] = ["missing.jsonl"]
        self.write_config(config)

        with self.assertRaisesRegex(MODULE.ConfigError, "does not exist"):
            MODULE.load_config(self.config_path)

    def test_invalid_model_dimension_is_rejected(self):
        config = self.config()
        config["model"]["n_layer"] = 0
        self.write_config(config)

        with self.assertRaises(MODULE.ConfigError):
            MODULE.load_config(self.config_path)

    def test_evaluation_requires_three_sessions(self):
        config = self.config()
        config["evaluation"] = {
            "enabled": True,
            "output": "out/evaluation.json",
            "k": 1,
        }
        self.write_config(config)

        with self.assertRaisesRegex(MODULE.ConfigError, "at least 3"):
            MODULE.load_config(self.config_path)

    def test_existing_output_is_never_overwritten(self):
        config = self.config()
        output = self.base / "out/profile.json"
        output.parent.mkdir()
        output.write_text("preserve", encoding="utf-8")
        self.write_config(config)

        with self.assertRaisesRegex(MODULE.ConfigError, "Refusing to overwrite"):
            MODULE.load_config(self.config_path)

        self.assertEqual(output.read_text(), "preserve")

    def test_output_cannot_overwrite_input(self):
        config = self.config()
        config["profile"]["output"] = "session.jsonl"
        self.write_config(config)

        with self.assertRaisesRegex(MODULE.ConfigError, "overwrite an input"):
            MODULE.load_config(self.config_path)

    @patch.object(MODULE.subprocess, "run")
    def test_execute_rejects_nonzero_exit_code(self, mocked_run):
        mocked_run.return_value.returncode = 7

        with self.assertRaisesRegex(RuntimeError, "exit code 7"):
            MODULE.execute(["fake-command"], "Test command")

    @patch.object(MODULE.subprocess, "run")
    def test_execute_accepts_zero_exit_code(self, mocked_run):
        mocked_run.return_value.returncode = 0

        self.assertIsNone(
            MODULE.execute(["fake-command"], "Test command")
        )



    def test_evaluation_failure_rolls_back_created_outputs(self):
        config = self.config()
        config["inputs"] = [
            "session.jsonl", "session2.jsonl", "session3.jsonl"
        ]
        for name in config["inputs"][1:]:
            (self.base / name).write_text("{}\\n", encoding="utf-8")

        config["evaluation"] = {
            "enabled": True,
            "output": "out/evaluation.json",
            "k": 1,
        }
        self.write_config(config)

        profile = self.base / "out/profile.json"
        evaluation = self.base / "out/evaluation.json"
        summary = self.base / "out/pwea_run_summary.json"
        checksums = self.base / "out/PWEA_RUN_SHA256SUMS.txt"

        def fake_subprocess(command, check=False):
            if "build_profile.py" in command[1]:
                output = Path(command[command.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text('{"partial": true}', encoding="utf-8")
                return SimpleNamespace(returncode=0)
            return SimpleNamespace(returncode=9)

        with patch.object(
            MODULE.sys, "argv", ["run.py", "--config", str(self.config_path)]
        ), patch.object(
            MODULE.subprocess, "run", side_effect=fake_subprocess
        ):
            result = MODULE.main()

        self.assertEqual(result, 2)
        self.assertFalse(profile.exists())
        self.assertFalse(evaluation.exists())
        self.assertFalse(summary.exists())
        self.assertFalse(checksums.exists())
        self.assertTrue(self.input_path.exists())



if __name__ == "__main__":
    unittest.main()
