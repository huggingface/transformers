import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("run_mic", Path(__file__).resolve().parents[2] / "utils/run_mic.py")
run_mic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_mic)


class RunMicTest(unittest.TestCase):
    def test_git_selection_includes_branch_and_working_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return run_mic.git(root, *args)

            git("init", "-q")
            git("config", "user.email", "test@example.com")
            git("config", "user.name", "Test")
            for name in ("renamed", "deleted", "unstaged"):
                (root / name).write_text("original")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            (root / "committed").write_text("branch change")
            git("add", ".")
            git("commit", "-qm", "branch")
            git("mv", "renamed", "destination")
            (root / "deleted").unlink()
            (root / "unstaged").write_text("edited")
            (root / "untracked").write_text("new")
            self.assertEqual(
                run_mic.changed_files(root, base),
                ["committed", "deleted", "destination", "renamed", "unstaged", "untracked"],
            )

    def test_model_uses_selector_local_source_and_propagates_failure(self):
        def execute(command, cwd, env):
            self.assertEqual(cwd, run_mic.ROOT)
            self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(run_mic.ROOT / "src"))
            self.assertIn("--fixtures-only", command)
            self.assertIn("--fleet", command)
            path = Path(command[command.index("--changed-files") + 1])
            self.assertEqual(path.read_text(encoding="utf-8"), "src/transformers/models/bert/__init__.py\n")
            return subprocess.CompletedProcess(command, 1)

        with (
            patch.dict(os.environ, {"MODEL": "bert", "ALL": ""}),
            patch.object(run_mic.importlib.util, "find_spec", return_value=object()),
            patch.object(run_mic.subprocess, "run", side_effect=execute),
        ):
            self.assertEqual(run_mic.main(), 1)

    def test_clean_checkout_does_not_require_runner(self):
        with (
            patch.dict(os.environ, {"MODEL": "", "ALL": ""}),
            patch.object(run_mic, "changed_files", return_value=[]),
            patch.object(run_mic.importlib.util, "find_spec") as find_spec,
        ):
            self.assertEqual(run_mic.main(), 0)
            find_spec.assert_not_called()

    def test_unknown_model_fails(self):
        with patch.dict(os.environ, {"MODEL": "../bert", "ALL": ""}):
            self.assertEqual(run_mic.main(), 2)
