import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("check_mic", Path(__file__).resolve().parents[2] / "utils/check_mic.py")
check_mic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_mic)


class CheckMicTest(unittest.TestCase):
    def test_cached_snapshot_is_verified_offline_against_the_fleet_hashes(self):
        from transformers_mic import resources

        sys.path.insert(0, str(resources.backend()))
        import run as mic_run

        with tempfile.TemporaryDirectory() as cache:
            folder = Path(cache) / "models--org--fixture" / "snapshots" / "abc"
            folder.mkdir(parents=True)
            (folder / "config.json").write_text("{}")
            (folder / "model.safetensors").write_text("weights")
            files = mic_run.file_hashes(folder)
            (folder / "build.json").write_text(json.dumps({"sha256": files}))
            (folder / "integration.json").write_text("{}")
            pins = [
                ("contract", {"files_sha256": mic_run.digest(mic_run.contract_files(folder))}),
                ("fixture", {"files_sha256": mic_run.digest(files)}),
            ]
            with patch("huggingface_hub.constants.HF_HUB_CACHE", cache):
                self.assertTrue(check_mic.cached_snapshot("org/fixture", "abc", pins))
                # Another revision, or a pin the cached files do not match, is not cached.
                self.assertFalse(check_mic.cached_snapshot("org/fixture", "def", pins))
                self.assertFalse(check_mic.cached_snapshot("org/fixture", "abc", [("fixture", {"files_sha256": "x"})]))
                # A partial snapshot (weights never fetched) is not cached either.
                (folder / "model.safetensors").unlink()
                self.assertFalse(check_mic.cached_snapshot("org/fixture", "abc", pins[1:]))

    def test_git_selection_includes_branch_and_working_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return check_mic.git(root, *args)

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
                check_mic.changed_files(root, base),
                ["committed", "deleted", "destination", "renamed", "unstaged", "untracked"],
            )

    def test_model_runs_only_its_contracts_and_propagates_failure(self):
        def execute(command, env, fleet_name, total=None, jobs=1):
            self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(check_mic.ROOT / "src"))
            self.assertIn("--fixtures-only", command)
            self.assertIn("--fleet", command)
            self.assertEqual(command[command.index("--fleet") + 1], "prototypes")
            self.assertEqual(fleet_name, "prototypes")
            self.assertNotIn("--changed-files", command)
            self.assertNotIn("--run-all", command)
            self.assertEqual(command[command.index("--contract") + 1], "BERT")
            return 1

        with (
            patch.dict(os.environ, {"MODEL": "bert", "ALL": "", "FLEET": ""}),
            patch.object(check_mic.importlib.util, "find_spec", return_value=object()),
            patch.object(check_mic, "model_contracts", return_value=["BERT"]) as model_contracts,
            patch.object(check_mic, "fleet_contracts", return_value={"BERT": {}}),
            patch.object(check_mic, "download_fixtures"),
            patch.object(check_mic, "supports_allow_missing_baseline", return_value=False),
            patch.object(check_mic, "run", side_effect=execute),
        ):
            self.assertEqual(check_mic.main(), 1)
            model_contracts.assert_called_once_with(check_mic.ROOT, "bert", "prototypes")

    def test_model_without_contract_runs_nothing(self):
        with (
            patch.dict(os.environ, {"MODEL": "gpt2", "ALL": ""}),
            patch.object(check_mic.importlib.util, "find_spec", return_value=object()),
            patch.object(check_mic, "model_contracts", return_value=[]),
            patch.object(check_mic, "run") as execute,
        ):
            self.assertEqual(check_mic.main(), 0)
            execute.assert_not_called()

    def test_clean_checkout_does_not_require_runner(self):
        with (
            patch.dict(os.environ, {"MODEL": "", "ALL": ""}),
            patch.object(check_mic, "changed_files", return_value=[]),
            patch.object(check_mic.importlib.util, "find_spec") as find_spec,
        ):
            self.assertEqual(check_mic.main(), 0)
            find_spec.assert_not_called()

    def test_unknown_model_fails(self):
        with patch.dict(os.environ, {"MODEL": "../bert", "ALL": ""}):
            self.assertEqual(check_mic.main(), 2)

    def test_output_digest(self):
        records = [
            {"fleet_repository": "huggingface-mic/transformers-fleet", "fleet_revision": "276e4bec9df494de9d35"},
            {"selection": {"BERT": "models/bert (src/transformers/models/bert/modeling_bert.py)"}, "code_dirs": {}},
            {
                "contract": "r",
                "contract_name": "BERT",
                "variant": "default",
                "target": "slim",
                "status": "pass",
                "seconds": 1.6,
                "failures": [],
            },
            {
                "contract": "r",
                "contract_name": "CLIP",
                "variant": "sdpa",
                "target": "slim",
                "status": "fail",
                "seconds": 2.0,
                "failures": ["logits differ"],
                "environment_differs": {"torch": "2.14 (pinned 2.11)"},
            },
            {"summary": {"pass": 1, "fail": 1}, "seconds": 3.6, "isolate": False},
        ]
        script = "import sys\n" + "".join(f"print({json.dumps(json.dumps(r))})\n" for r in records)
        script += "print('noise')\nsys.exit(1)\n"
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(check_mic.run([sys.executable, "-c", script], os.environ.copy()), 1)
        lines = stdout.getvalue().splitlines()
        self.assertEqual(lines[0], "Fleet prototypes (huggingface-mic/transformers-fleet@276e4bec9d)")
        # The runner reports a contract only when it finishes, so the wait is announced.
        self.assertEqual(lines[1], "Running the selected contracts, results print as each one finishes...")
        self.assertEqual(lines[2], "Selected BERT: models/bert (src/transformers/models/bert/modeling_bert.py)")
        self.assertEqual(lines[3], "✓ BERT  slim  pass  1.6s")
        self.assertEqual(lines[4], "✗ CLIP [sdpa]  slim  fail  2.0s")
        self.assertEqual(lines[5], "    logits differ")
        self.assertIn("torch 2.14 (pinned 2.11)", lines[6])
        # A failing target explains the exit code; raw runner noise stays hidden.
        self.assertNotIn("noise", stdout.getvalue())
        self.assertRegex(lines[-1], r"^MIC: 1 passed, 1 failed in [0-9.]+s$")

    def test_progress_counts_contracts_and_beats_while_the_runner_is_quiet(self):
        record = {"contract": "r", "contract_name": "BERT", "target": "slim", "status": "pass", "failures": []}
        script = (
            "import json, time\n"
            "time.sleep(1.5)\n"
            f"print(json.dumps({record!r}))\n"
            f"print(json.dumps({ {**record, 'target': 'other'}!r}))\n"
        )
        with patch.object(check_mic, "HEARTBEAT_SECONDS", 1), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(check_mic.run([sys.executable, "-c", script], os.environ.copy(), total=2), 0)
        lines = stdout.getvalue().splitlines()
        self.assertRegex(lines[0], r"^\.\.\. 1s in, 0/2 contracts done, loading the first contract$")
        # Both targets belong to one contract: the counter counts contracts.
        self.assertEqual(lines[1], "[1/2] ✓ BERT  slim  pass")
        self.assertEqual(lines[2], "[1/2] ✓ BERT  other  pass")

    def test_contracts_are_split_over_parallel_runners(self):
        # A fake runner: announces the fleet, then passes every --contract it was given.
        script = (
            "import json, os, sys\n"
            "names = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == '--contract']\n"
            "print(json.dumps({'fleet_repository': 'f', 'fleet_revision': '0123456789abc'}))\n"
            "for name in names:\n"
            "    print(json.dumps({'contract': 'r', 'contract_name': name, 'target': os.environ['OMP_NUM_THREADS'],"
            " 'status': 'pass', 'failures': []}))\n"
        )
        command = [sys.executable, "-c", script] + [a for n in ("A", "B", "C") for a in ("--contract", n)]
        self.assertEqual(len(check_mic.shard(command, 2)), 2)
        self.assertEqual(check_mic.shard(command, 1), [command])
        env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
        with patch("sys.stdout", new_callable=io.StringIO) as stdout, patch("os.cpu_count", return_value=8):
            self.assertEqual(check_mic.run(command, env, total=3, jobs=2), 0)
        output = stdout.getvalue()
        self.assertEqual(output.count("Fleet prototypes (f@0123456789)"), 1)
        self.assertIn("Running 3 contracts in 2 parallel runners", output)
        # Every contract ran once, each runner with its share of the CPU threads.
        self.assertEqual(sorted(line.split()[2] for line in output.splitlines() if "✓" in line), ["A", "B", "C"])
        self.assertEqual({line.split()[3] for line in output.splitlines() if "✓" in line}, {"4"})
        self.assertRegex(output.splitlines()[-1], r"^MIC: 3 passed in [0-9.]+s$")

    def test_many_contracts_show_a_bar_and_only_the_failures(self):
        records = [
            {"contract": "r", "contract_name": f"C{i}", "target": "t", "status": "pass", "failures": []}
            for i in range(12)
        ]
        records[5] = {**records[5], "status": "fail", "failures": ["logits differ"]}
        script = "".join(f"print({json.dumps(json.dumps(r))})\n" for r in records)
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(check_mic.run([sys.executable, "-c", script], os.environ.copy(), total=12), 1)
        lines = stdout.getvalue().rstrip("\n").split("\n")  # not splitlines(): it splits on "\r" too
        # Every line after the first bar redraws it, except the failure (printed in full) and the bar after it.
        self.assertEqual(lines[0], "███" + "░" * 37 + " 1/12")
        self.assertEqual(lines[5], "\r✗ C5  t  fail")
        self.assertEqual(lines[6], "    logits differ")
        self.assertEqual(lines[7], "████████████████████" + "░" * 20 + " 6/12")
        self.assertEqual(lines[8], "\r" + "███████████████████████" + "░" * 17 + " 7/12")
        self.assertEqual(lines[-2], "\r" + "█" * 40 + " 12/12")
        self.assertNotIn("C0", stdout.getvalue())
        self.assertRegex(lines[-1], r"^MIC: 11 passed, 1 failed in [0-9.]+s$")

    def test_install_hints_name_what_is_missing(self):
        import builtins

        real_import = builtins.__import__

        def released_doc_builder(name, *args, **kwargs):
            # The released hf-doc-builder has no `doc_builder.testing`.
            if name == "doc_builder.testing":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        no_mic = patch("importlib.util.find_spec", return_value=None)
        old_doc_builder = patch("builtins.__import__", side_effect=released_doc_builder)
        with patch("importlib.util.find_spec", return_value=object()), old_doc_builder:
            self.assertEqual(check_mic.missing_requirements(), [check_mic.DOC_BUILDER_HINT])
        with no_mic:
            hints = check_mic.missing_requirements()
        self.assertEqual(hints[0], check_mic.MIC_HINT)
        self.assertIn('pip install -e ".[testing]"', check_mic.MIC_HINT)
        self.assertIn("git+https://github.com/huggingface/doc-builder.git", check_mic.DOC_BUILDER_HINT)

    def test_runner_crash_shows_raw_tail(self):
        script = "import sys\nprint('Traceback: boom')\nsys.exit(2)\n"
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(check_mic.run([sys.executable, "-c", script], os.environ.copy()), 2)
        self.assertIn("Runner output:\n  Traceback: boom", stdout.getvalue())
        self.assertIn("MIC: nothing ran", stdout.getvalue())

    def test_unknown_fleet_fails(self):
        with patch.dict(os.environ, {"MODEL": "bert", "ALL": "", "FLEET": "nightly"}):
            self.assertEqual(check_mic.main(), 2)

    def test_missing_platform_baseline_reruns_without_comparison(self):
        fake_runner = """
import json, sys
calls = open(sys.argv[1], "a")
calls.write(" ".join(sys.argv[2:]) + "\\n")
record = {"contract": "r", "contract_name": "DistilBERT", "variant": "default", "target": "slim", "platform": "cpu-arm64"}
if "--no-baseline" in sys.argv:
    print(json.dumps({**record, "status": "pass", "failures": []}))
    sys.exit(0)
print(json.dumps({**record, "status": "fail", "failures": ["no reviewed baseline at /x/cpu-arm64/slim.json"]}))
print(json.dumps({**record, "contract_name": "BERT", "status": "pass", "failures": []}))
sys.exit(1)
"""
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "runner.py"
            script.write_text(fake_runner, encoding="utf-8")
            calls = Path(directory) / "calls"
            command = [sys.executable, str(script), str(calls), "--fleet", "prototypes", "--run-all"]
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(check_mic.run(command, os.environ.copy()), 0)
            self.assertEqual(
                calls.read_text(encoding="utf-8").splitlines(),
                ["--fleet prototypes --run-all", "--fleet prototypes --contract DistilBERT --no-baseline"],
            )
        output = stdout.getvalue()
        self.assertNotIn("✗", output)
        self.assertIn("No reviewed cpu-arm64 baseline for DistilBERT; running without comparison.", output)
        self.assertIn("✓ DistilBERT  slim  pass (no cpu-arm64 baseline)", output)
        self.assertRegex(output.splitlines()[-1], r"^MIC: 1 passed, 1 ran without a baseline in [0-9.]+s$")

    def test_runner_flag_for_missing_baseline_is_reported_in_one_pass(self):
        record = {
            "contract": "r",
            "contract_name": "DistilBERT",
            "variant": "default",
            "target": "slim",
            "platform": "cpu-arm64",
            "status": "pass",
            "failures": [],
            "baseline_missing": True,
        }
        script = f"print({json.dumps(json.dumps(record))})\n"
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(check_mic.run([sys.executable, "-c", script], os.environ.copy()), 0)
        output = stdout.getvalue()
        self.assertIn("✓ DistilBERT  slim  pass (no cpu-arm64 baseline)", output)
        self.assertNotIn("running without comparison", output)
        self.assertRegex(output.splitlines()[-1], r"^MIC: 1 ran without a baseline in [0-9.]+s$")

    def test_whole_fleet_change_needs_all(self):
        with (
            patch.dict(os.environ, {"MODEL": "", "ALL": "", "FLEET": ""}),
            patch.object(check_mic, "changed_files", return_value=["setup.py"]),
            patch.object(check_mic.importlib.util, "find_spec", return_value=object()),
            patch.object(check_mic, "supports_allow_missing_baseline", return_value=False),
            patch.object(
                check_mic,
                "changed_contracts",
                return_value=(
                    {},
                    ["dependencies (setup.py)", "fleet revision (tests/contracts.lock)"],
                    {"BERT": {}, "CLIP": {}},
                ),
            ),
            patch.object(check_mic, "run") as execute,
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            self.assertEqual(check_mic.main(), 0)
        execute.assert_not_called()
        self.assertIn(
            "select every contract in the prototypes fleet (2):\n  dependencies (setup.py)\n"
            "  fleet revision (tests/contracts.lock)\n",
            stdout.getvalue(),
        )
        self.assertIn("ALL=1", stdout.getvalue())

    def test_model_directory_change_runs_selected_contracts(self):
        selection = {"DistilBERT": "models/distilbert (src/transformers/models/distilbert/modeling_distilbert.py)"}

        def execute(command, env, fleet_name, total=None, jobs=1):
            self.assertEqual(command[command.index("--contract") + 1], "DistilBERT")
            self.assertNotIn("--changed-files", command)
            return 0

        with (
            patch.dict(os.environ, {"MODEL": "", "ALL": "", "FLEET": ""}),
            patch.object(check_mic, "changed_files", return_value=list(selection)),
            patch.object(check_mic.importlib.util, "find_spec", return_value=object()),
            patch.object(check_mic, "supports_allow_missing_baseline", return_value=False),
            patch.object(check_mic, "changed_contracts", return_value=(selection, [], {"DistilBERT": {}})),
            patch.object(check_mic, "download_fixtures") as download,
            patch.object(check_mic, "run", side_effect=execute),
        ):
            self.assertEqual(check_mic.main(), 0)
        download.assert_called_once_with({"DistilBERT": {}}, ["DistilBERT"])
