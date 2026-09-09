# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from unittest.mock import patch


git_repo_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
utils_path = os.path.join(git_repo_path, "utils")
if utils_path not in sys.path:
    sys.path.append(utils_path)

import check_noisy_comments  # noqa: E402


LICENSE_HEADER = """# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""


NOISY_SOURCE = """def foo():
    # This is a long agent-style explanation that goes on and on.
    # It restates the code below in prose.
    # And keeps going with more detail.
    # Still going, adding nothing new.
    # One more line for good measure.
    # And a sixth line to push it over the limit.
    return 1
"""


class _StubOwners:
    """Stands in for `FileOwners`, so ownership tests do not need the reviewer resolver installed."""

    def __init__(self, logins):
        self.logins = logins

    def logins_for(self, path):
        return self.logins


class _StubResolver:
    CODEOWNERS_PATH = "codeowners"

    def __init__(self, source, owners):
        self.source = source
        self.owners = owners

    def resolution_source(self, file_path, codeowners_lines):
        return self.source

    def owners_for_file(self, file_path, codeowners_lines):
        return self.owners


class NoisyCommentsTest(unittest.TestCase):
    def _write_file(self, repo_root: Path, content: str) -> Path:
        path = repo_root / "sample.py"
        path.write_text(content, encoding="utf-8")
        return path

    def _blame(self, commit_date: date, author_email: str = "someone@example.com") -> check_noisy_comments.LineBlame:
        return check_noisy_comments.LineBlame(commit_date=commit_date, author_email=author_email)

    def _finding(self, path: Path, line: int = 2, end_line: int = 2) -> check_noisy_comments.Finding:
        return check_noisy_comments.Finding(
            path=path, line=line, end_line=end_line, code="NC001", message="", text="", score=1
        )

    def test_ignores_standard_license_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(repo_root, LICENSE_HEADER + "\nvalue = 1\n")

            with patch.object(check_noisy_comments, "ROOT", repo_root):
                findings = check_noisy_comments.check_file(path, max_block_lines=5, max_block_chars=500)

            self.assertEqual(findings, [])

    def test_ignores_leading_comments_before_first_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(
                repo_root,
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "",
                        "# coding=utf-8",
                        "# Copyright 2020 The HuggingFace Inc. team.",
                        "#",
                        '# Licensed under the Apache License, Version 2.0 (the "License");',
                        "# you may not use this file except in compliance with the License.",
                        "# You may obtain a copy of the License at",
                        "#",
                        "#     http://www.apache.org/licenses/LICENSE-2.0",
                        "#",
                        "# Unless required by applicable law or agreed to in writing, software",
                        '# distributed under the License is distributed on an "AS IS" BASIS,',
                        "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
                        "# See the License for the specific language governing permissions and",
                        "# limitations under the License.",
                        "",
                        "# this script dumps information about the environment",
                        "",
                        "import sys",
                    ]
                ),
            )

            with patch.object(check_noisy_comments, "ROOT", repo_root):
                findings = check_noisy_comments.check_file(path, max_block_lines=5, max_block_chars=500)

            self.assertEqual(findings, [])

    def test_flags_long_comment_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(
                repo_root,
                "\n".join(
                    [
                        "def foo():",
                        "    value = 1",
                        "    # This is a multi-line note.",
                        "    # It keeps going.",
                        "    # And going.",
                        "    # And going.",
                        "    # And going.",
                        "    # And going.",
                        "    return value",
                    ]
                ),
            )

            with patch.object(check_noisy_comments, "ROOT", repo_root):
                findings = check_noisy_comments.check_file(path, max_block_lines=5, max_block_chars=500)

            self.assertEqual([finding.code for finding in findings], ["NC001"])

    def test_ignores_module_level_comment_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(
                repo_root,
                "\n".join(
                    [
                        "value = 1",
                        "# This is a module-level note.",
                        "# It keeps going.",
                        "# And going.",
                        "# And going.",
                        "# And going.",
                        "# And going.",
                    ]
                ),
            )

            with patch.object(check_noisy_comments, "ROOT", repo_root):
                findings = check_noisy_comments.check_file(path, max_block_lines=5, max_block_chars=500)

            self.assertEqual(findings, [])

    def test_ignores_inline_script_metadata_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(
                repo_root,
                "\n".join(
                    [
                        "# /// script",
                        "# dependencies = [",
                        '#     "torch",',
                        '#     "torchaudio",',
                        "# ]",
                        "# ///",
                        "value = 1",
                    ]
                ),
            )

            with patch.object(check_noisy_comments, "ROOT", repo_root):
                findings = check_noisy_comments.check_file(path, max_block_lines=5, max_block_chars=500)

            self.assertEqual(findings, [])

    def test_skips_autogenerated_modular_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            model_dir = repo_root / "src" / "transformers" / "models" / "demo"
            model_dir.mkdir(parents=True)
            generated_file = model_dir / "modeling_demo.py"
            generated_file.write_text(
                "\n".join(
                    [
                        "# coding=utf-8",
                        "# This file was automatically generated from src/transformers/models/demo/modular_demo.py.",
                        "# Do not edit this file manually.",
                        "def foo():",
                        "    # one",
                        "    # two",
                        "    # three",
                        "    # four",
                        "    # five",
                        "    # six",
                    ]
                ),
                encoding="utf-8",
            )
            modular_file = model_dir / "modular_demo.py"
            modular_file.write_text(
                NOISY_SOURCE,
                encoding="utf-8",
            )

            with patch.object(check_noisy_comments, "ROOT", repo_root):
                findings = check_noisy_comments.check_comments(targets=["src"], max_block_lines=5)

            self.assertEqual([finding.path for finding in findings], [modular_file])

    def test_thresholds_are_configurable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(
                repo_root, "def foo():\n    # This comment is deliberately not short.\n    return 1\n"
            )

            with patch.object(check_noisy_comments, "ROOT", repo_root):
                strict = check_noisy_comments.check_file(path, max_block_lines=5, max_block_chars=10)
                relaxed = check_noisy_comments.check_file(path, max_block_lines=5, max_block_chars=100)

            self.assertEqual([finding.code for finding in strict], ["NC002"])
            self.assertEqual(relaxed, [])

    def test_cli_reports_without_failing_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(repo_root, NOISY_SOURCE)
            stdout = io.StringIO()

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_running_in_pr", return_value=False),
                patch.object(sys, "argv", ["check_noisy_comments.py", str(path), "--no-cache", "--progress", "never"]),
                redirect_stdout(stdout),
            ):
                exit_code = check_noisy_comments.main()

            self.assertEqual(exit_code, 0)
            self.assertIn("Reporting only; not blocking.", stdout.getvalue())

    def test_cli_path_option_checks_specific_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            checked_dir = repo_root / "checked"
            ignored_dir = repo_root / "ignored"
            checked_dir.mkdir()
            ignored_dir.mkdir()
            checked_file = checked_dir / "sample.py"
            checked_file.write_text(NOISY_SOURCE, encoding="utf-8")
            ignored_file = ignored_dir / "sample.py"
            ignored_file.write_text(
                "def foo():\n    # This comment is deliberately not short.\n    return 1\n", encoding="utf-8"
            )
            stdout = io.StringIO()

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_running_in_pr", return_value=False),
                patch.object(
                    sys, "argv", ["check_noisy_comments.py", "--path", "checked", "--no-cache", "--progress", "never"]
                ),
                redirect_stdout(stdout),
            ):
                exit_code = check_noisy_comments.main()

            self.assertEqual(exit_code, 0)
            self.assertIn("checked/sample.py", stdout.getvalue())
            self.assertNotIn("ignored/sample.py", stdout.getvalue())

    def test_collect_findings_uses_persistent_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            cache_path = repo_root / "utils" / ".noisy_comments_cache.json"
            cache_path.parent.mkdir()
            path = self._write_file(repo_root, NOISY_SOURCE)

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "CACHE_PATH", cache_path),
                patch.object(check_noisy_comments, "_file_line_blames", return_value={}),
            ):
                first = check_noisy_comments.collect_findings(targets=[str(path)], use_cache=True)
                with patch.object(check_noisy_comments, "check_file", side_effect=AssertionError("cache miss")):
                    second = check_noisy_comments.collect_findings(targets=[str(path)], use_cache=True)

            self.assertEqual([finding.code for finding in first], ["NC001"])
            self.assertEqual([finding.code for finding in second], ["NC001"])
            self.assertTrue(cache_path.exists())

    def test_collect_findings_shows_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(repo_root, NOISY_SOURCE)
            stdout = io.StringIO()

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_file_line_blames", return_value={}),
                redirect_stdout(stdout),
            ):
                check_noisy_comments.collect_findings(targets=[str(path)], use_cache=False, progress=True)

            self.assertIn("\rScanning [", stdout.getvalue())
            self.assertIn("Scanned [", stdout.getvalue())
            self.assertIn("] 1/1", stdout.getvalue())
            self.assertEqual(stdout.getvalue().count("\n"), 1)

    def test_pr_scope_filters_to_changed_python_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            changed_file = repo_root / "changed.py"
            changed_file.write_text(NOISY_SOURCE, encoding="utf-8")
            unchanged_file = repo_root / "unchanged.py"
            unchanged_file.write_text(NOISY_SOURCE, encoding="utf-8")

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_patch_added_lines", return_value={changed_file: {2}}),
                patch.object(check_noisy_comments, "_file_line_blames", return_value={}),
            ):
                findings = check_noisy_comments.collect_findings(
                    targets=[str(repo_root)], use_cache=False, diff_only=True
                )

            self.assertEqual([finding.path for finding in findings], [changed_file])

    def test_cli_orders_biggest_offenders_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            large_file = repo_root / "large.py"
            large_file.write_text("def foo():\n    # " + "a" * 600 + "\n    return 1\n", encoding="utf-8")
            small_file = repo_root / "small.py"
            small_file.write_text(NOISY_SOURCE, encoding="utf-8")
            stdout = io.StringIO()

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_running_in_pr", return_value=False),
                patch.object(
                    sys,
                    "argv",
                    ["check_noisy_comments.py", "--path", str(repo_root), "--no-cache", "--progress", "never"],
                ),
                redirect_stdout(stdout),
            ):
                exit_code = check_noisy_comments.main()

            self.assertEqual(exit_code, 0)
            self.assertLess(stdout.getvalue().index("large.py"), stdout.getvalue().index("small.py"))

    def test_cli_filters_by_rule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            # Tall enough for NC001 and long enough for NC002, so `--rule` has something to pick from.
            block = "\n".join(f"    # {'detail ' * 15}" for _ in range(6))
            path = self._write_file(repo_root, f"def foo():\n{block}\n    return 1\n")
            stdout = io.StringIO()

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_running_in_pr", return_value=False),
                patch.object(
                    sys,
                    "argv",
                    [
                        "check_noisy_comments.py",
                        "--path",
                        str(path),
                        "--rule",
                        "NC002",
                        "--no-cache",
                        "--progress",
                        "never",
                    ],
                ),
                redirect_stdout(stdout),
            ):
                exit_code = check_noisy_comments.main()

            self.assertEqual(exit_code, 0)
            self.assertIn("NC002", stdout.getvalue())
            self.assertNotIn("NC001", stdout.getvalue())

    def test_cutoff_filter_ignores_old_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(repo_root, NOISY_SOURCE)
            finding = check_noisy_comments.Finding(
                path=path,
                line=2,
                end_line=2,
                code="NC001",
                message="",
                text="",
                score=1,
            )

            blames = {2: self._blame(date(2024, 12, 31))}
            with patch.object(check_noisy_comments, "_file_line_blames", return_value=blames):
                findings = check_noisy_comments._filter_findings_by_cutoff([finding], date(2025, 1, 1))

            self.assertEqual(findings, [])

    def test_cutoff_filter_keeps_findings_on_cutoff_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(repo_root, NOISY_SOURCE)
            finding = check_noisy_comments.Finding(
                path=path,
                line=2,
                end_line=2,
                code="NC001",
                message="",
                text="",
                score=1,
            )

            blames = {2: self._blame(date(2025, 1, 1))}
            with patch.object(check_noisy_comments, "_file_line_blames", return_value=blames):
                findings = check_noisy_comments._filter_findings_by_cutoff([finding], date(2025, 1, 1))

            self.assertEqual(findings, [finding])

    def _filter_by_ownership(self, findings, owner_logins, blames):
        owners = _StubOwners(owner_logins)
        with patch.object(check_noisy_comments, "_file_line_blames", return_value=blames):
            return check_noisy_comments._filter_findings_by_ownership(findings, owners)

    def test_ownership_filter_ignores_comments_from_the_files_owner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_file(Path(tmpdir), NOISY_SOURCE)
            finding = self._finding(path)

            blames = {2: self._blame(date(2026, 6, 1), "cyril.vallez@huggingface.co")}
            findings = self._filter_by_ownership([finding], {"cyrilvallez"}, blames)

            self.assertEqual(findings, [])

    def test_ownership_filter_keeps_comments_from_a_non_owner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_file(Path(tmpdir), NOISY_SOURCE)
            finding = self._finding(path)

            blames = {2: self._blame(date(2026, 6, 1), "someone@example.com")}
            findings = self._filter_by_ownership([finding], {"cyrilvallez"}, blames)

            self.assertEqual(findings, [finding])

    def test_ownership_filter_keeps_comments_from_an_owner_of_another_area(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_file(Path(tmpdir), NOISY_SOURCE)
            finding = self._finding(path)

            # Owning `generation/` does not make a comment in someone else's file deliberate.
            blames = {2: self._blame(date(2026, 6, 1), "cyril.vallez@huggingface.co")}
            findings = self._filter_by_ownership([finding], {"sunmarc"}, blames)

            self.assertEqual(findings, [finding])

    def test_ownership_filter_keeps_blocks_with_a_non_owner_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_file(Path(tmpdir), "def foo():\n    # one\n    # two\n    return 1\n")
            finding = self._finding(path, line=2, end_line=3)

            blames = {
                2: self._blame(date(2026, 6, 1), "cyril.vallez@huggingface.co"),
                3: self._blame(date(2026, 6, 2), "someone@example.com"),
            }
            findings = self._filter_by_ownership([finding], {"cyrilvallez"}, blames)

            self.assertEqual(findings, [finding])

    def test_ownership_filter_keeps_findings_in_unowned_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_file(Path(tmpdir), NOISY_SOURCE)
            finding = self._finding(path)

            blames = {2: self._blame(date(2026, 6, 1), "cyril.vallez@huggingface.co")}
            findings = self._filter_by_ownership([finding], set(), blames)

            self.assertEqual(findings, [finding])

    def test_ownership_filter_keeps_findings_without_blame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_file(Path(tmpdir), NOISY_SOURCE)
            finding = self._finding(path)

            findings = self._filter_by_ownership([finding], {"cyrilvallez"}, {})

            self.assertEqual(findings, [finding])

    def test_author_logins_derives_logins_from_commit_email(self):
        self.assertIn(
            "arthurzucker", check_noisy_comments._author_logins("48595927+ArthurZucker@users.noreply.github.com")
        )
        self.assertIn("cyrilvallez", check_noisy_comments._author_logins("cyril.vallez@huggingface.co"))
        self.assertIn("rocketknight1", check_noisy_comments._author_logins("rocketknight1@gmail.com"))
        # An owner whose commit address resembles neither their login nor their name needs an alias.
        self.assertIn("zucchininlp", check_noisy_comments._author_logins("raushan@huggingface.co"))
        self.assertEqual(check_noisy_comments._author_logins(""), set())

    def test_author_logins_does_not_match_lookalike_contributors(self):
        # Both of these are real contributors in this history; neither is the owner they resemble.
        self.assertNotIn("cyrilvallez", check_noisy_comments._author_logins("cyrile.ufr.orsay@gmail.com"))
        self.assertNotIn("vasqu", check_noisy_comments._author_logins("lmvasque@users.noreply.github.com"))

    def test_file_owners_reads_the_resolver_and_ignores_the_catch_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            codeowners = repo_root / "codeowners"
            codeowners.write_text("/src/ @Someone\n", encoding="utf-8")
            owned = self._write_file(repo_root, "value = 1\n")

            resolver = _StubResolver(source="rule", owners=["@SomeOne", "@other"])
            lines = codeowners.read_text(encoding="utf-8").splitlines(keepends=True)
            with patch.object(check_noisy_comments, "ROOT", repo_root):
                owners = check_noisy_comments.FileOwners(resolver, lines)
                self.assertEqual(owners.logins_for(owned), {"someone", "other"})

                resolver.source = "catch-all"
                self.assertEqual(check_noisy_comments.FileOwners(resolver, lines).logins_for(owned), set())

    def test_blame_parsing_reads_author_email_and_date(self):
        porcelain = (
            "abc123 1 1 1\n"
            "author Cyril Vallez\n"
            "author-mail <cyril.vallez@huggingface.co>\n"
            "author-time 1767225600\n"
            "\tdef foo():\n"
            "def456 2 2 1\n"
            "author Someone Else\n"
            "author-mail <someone@example.com>\n"
            "author-time 1767312000\n"
            "\t    return 1\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(repo_root, "def foo():\n    return 1\n")

            completed = type("Completed", (), {"returncode": 0, "stdout": porcelain})()
            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_FILE_LINE_BLAME_CACHE", {}),
                patch.object(check_noisy_comments.subprocess, "run", return_value=completed),
            ):
                blames = check_noisy_comments._file_line_blames(path)

            self.assertEqual(blames[1].author_email, "cyril.vallez@huggingface.co")
            self.assertEqual(blames[2].author_email, "someone@example.com")
            self.assertEqual(blames[1].commit_date, date(2026, 1, 1))

    def test_noqa_marker_suppresses_only_the_named_rule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            block = "\n".join(f"    # line {index}" for index in range(6))
            path = self._write_file(repo_root, f"def foo():\n{block}\n    # noqa: NC001\n    return 1\n")

            with patch.object(check_noisy_comments, "ROOT", repo_root):
                findings = check_noisy_comments.check_file(path, max_block_lines=5, max_block_chars=500)

            self.assertEqual([finding.code for finding in findings], [])

            # The same block trips NC001 and NC002; the marker names one code, not all of them.
            block = "\n".join(f"    # {'detail ' * 15}" for _ in range(6))
            path = self._write_file(repo_root, f"def foo():\n{block}\n    # noqa: NC001\n    return 1\n")
            with patch.object(check_noisy_comments, "ROOT", repo_root):
                findings = check_noisy_comments.check_file(path, max_block_lines=5, max_block_chars=500)

            self.assertEqual([finding.code for finding in findings], ["NC002"])

    def test_patch_added_lines_parses_diff_hunks(self):
        diff = "\n".join(
            [
                "diff --git a/kept.py b/kept.py",
                "--- a/kept.py",
                "+++ b/kept.py",
                "@@ -10,0 +11,3 @@ def foo():",
                "+    # one",
                "+    # two",
                "+    # three",
                "@@ -40 +43 @@ def bar():",
                "+    # single",
                "diff --git a/notes.md b/notes.md",
                "--- a/notes.md",
                "+++ b/notes.md",
                "@@ -1,0 +2,1 @@",
                "+text",
            ]
        )
        with (
            patch.object(check_noisy_comments, "_PATCH_ADDED_LINES_CACHE", {}),
            patch.object(check_noisy_comments, "_git_output", side_effect=["abc123", diff]),
        ):
            added = check_noisy_comments._patch_added_lines()

        self.assertEqual(added, {check_noisy_comments.ROOT / "kept.py": {11, 12, 13, 43}})

    def test_patch_scope_keeps_only_findings_overlapping_added_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_file(Path(tmpdir), "def foo():\n    # one\n    # two\n    return 1\n")
            touched = self._finding(path, line=2, end_line=3)
            untouched = self._finding(path, line=20, end_line=20)

            # A block the patch only extends still counts: the patch is what pushed it over the limit.
            with patch.object(check_noisy_comments, "_patch_added_lines", return_value={path: {3}}):
                findings = check_noisy_comments._filter_findings_to_patch([touched, untouched])

            self.assertEqual(findings, [touched])

    def test_patch_scope_keeps_everything_when_the_diff_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_file(Path(tmpdir), "def foo():\n    # one\n    return 1\n")
            finding = self._finding(path)

            with patch.object(check_noisy_comments, "_patch_added_lines", return_value=None):
                self.assertEqual(check_noisy_comments._filter_findings_to_patch([finding]), [finding])

    def test_cli_blocks_on_findings_a_patch_adds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(repo_root, NOISY_SOURCE)
            argv = [
                "check_noisy_comments.py",
                str(path),
                "--no-cache",
                "--no-owner-filter",
                "--no-date-filter",
                "--progress",
                "never",
            ]

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_running_in_pr", return_value=True),
                patch.object(check_noisy_comments, "_patch_added_lines", return_value={path: {2}}),
                patch.object(sys, "argv", argv),
                redirect_stdout(io.StringIO()) as stdout,
            ):
                exit_code = check_noisy_comments.main()

            self.assertEqual(exit_code, 1)
            self.assertIn("on lines this patch adds. Blocking.", stdout.getvalue())
            self.assertIn("# noqa:", stdout.getvalue())

    def test_cli_ignores_findings_a_patch_did_not_add(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(repo_root, NOISY_SOURCE)
            argv = [
                "check_noisy_comments.py",
                str(path),
                "--no-cache",
                "--no-owner-filter",
                "--no-date-filter",
                "--progress",
                "never",
            ]

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_running_in_pr", return_value=True),
                patch.object(check_noisy_comments, "_patch_added_lines", return_value={path: {8}}),
                patch.object(sys, "argv", argv),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = check_noisy_comments.main()

            self.assertEqual(exit_code, 0)

    def test_cli_can_be_told_not_to_fail_in_pr_ci(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(repo_root, NOISY_SOURCE)
            argv = [
                "check_noisy_comments.py",
                str(path),
                "--no-cache",
                "--no-owner-filter",
                "--no-date-filter",
                "--no-fail-on-findings",
                "--progress",
                "never",
            ]

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_running_in_pr", return_value=True),
                patch.object(check_noisy_comments, "_patch_added_lines", return_value={path: {2}}),
                patch.object(sys, "argv", argv),
                redirect_stdout(io.StringIO()) as stdout,
            ):
                exit_code = check_noisy_comments.main()

            self.assertEqual(exit_code, 0)
            self.assertIn("Reporting only; not blocking.", stdout.getvalue())

    def test_cli_does_not_block_when_the_diff_cannot_be_resolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(repo_root, NOISY_SOURCE)
            argv = [
                "check_noisy_comments.py",
                str(path),
                "--no-cache",
                "--no-owner-filter",
                "--no-date-filter",
                "--progress",
                "never",
            ]

            # No merge base (a CI checkout without `origin/main`): the scan covers the whole tree, so
            # blocking would fail PRs for comments they never touched.
            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_running_in_pr", return_value=True),
                patch.object(check_noisy_comments, "_patch_added_lines", return_value=None),
                patch.object(sys, "argv", argv),
                redirect_stdout(io.StringIO()) as stdout,
            ):
                exit_code = check_noisy_comments.main()

            self.assertEqual(exit_code, 0)
            self.assertIn("Reporting only; not blocking.", stdout.getvalue())

    def test_cli_can_fail_on_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(repo_root, NOISY_SOURCE)

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_running_in_pr", return_value=False),
                patch.object(
                    sys,
                    "argv",
                    ["check_noisy_comments.py", str(path), "--fail-on-findings", "--no-cache", "--progress", "never"],
                ),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = check_noisy_comments.main()

            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
