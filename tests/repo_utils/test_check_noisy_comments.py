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


class NoisyCommentsTest(unittest.TestCase):
    def _write_file(self, repo_root: Path, content: str) -> Path:
        path = repo_root / "sample.py"
        path.write_text(content, encoding="utf-8")
        return path

    def test_ignores_standard_license_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(repo_root, LICENSE_HEADER + "\nvalue = 1\n")

            with patch.object(check_noisy_comments, "ROOT", repo_root):
                findings = check_noisy_comments.check_file(
                    path, max_block_lines=5, max_block_chars=500, max_comment_chars=500
                )

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
                findings = check_noisy_comments.check_file(
                    path, max_block_lines=5, max_block_chars=500, max_comment_chars=500
                )

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
                findings = check_noisy_comments.check_file(
                    path, max_block_lines=5, max_block_chars=500, max_comment_chars=500
                )

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
                findings = check_noisy_comments.check_file(
                    path, max_block_lines=5, max_block_chars=500, max_comment_chars=500
                )

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
                findings = check_noisy_comments.check_file(
                    path, max_block_lines=5, max_block_chars=500, max_comment_chars=500
                )

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
                        "value = 1",
                        '# Use the "simple path" when possible.',
                    ]
                ),
                encoding="utf-8",
            )
            modular_file = model_dir / "modular_demo.py"
            modular_file.write_text(
                'def foo():\n    # Use the "simple path" when possible.\n    return 1\n',
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
                strict = check_noisy_comments.check_file(
                    path, max_block_lines=5, max_block_chars=500, max_comment_chars=10
                )
                relaxed = check_noisy_comments.check_file(
                    path, max_block_lines=5, max_block_chars=500, max_comment_chars=100
                )

            self.assertEqual([finding.code for finding in strict], ["NC003"])
            self.assertEqual(relaxed, [])

    def test_flags_double_quoted_prose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(
                repo_root, 'def foo():\n    # Use the "simple path" when possible.\n    return 1\n'
            )

            with patch.object(check_noisy_comments, "ROOT", repo_root):
                findings = check_noisy_comments.check_file(
                    path, max_block_lines=5, max_block_chars=500, max_comment_chars=500
                )

            self.assertEqual([finding.code for finding in findings], ["NC004"])

    def test_cli_reports_without_failing_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(
                repo_root, 'def foo():\n    # Use the "simple path" when possible.\n    return 1\n'
            )
            stdout = io.StringIO()

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(sys, "argv", ["check_noisy_comments.py", str(path), "--no-cache"]),
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
            checked_file.write_text(
                'def foo():\n    # Use the "simple path" when possible.\n    return 1\n', encoding="utf-8"
            )
            ignored_file = ignored_dir / "sample.py"
            ignored_file.write_text(
                "def foo():\n    # This comment is deliberately not short.\n    return 1\n", encoding="utf-8"
            )
            stdout = io.StringIO()

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(sys, "argv", ["check_noisy_comments.py", "--path", "checked", "--no-cache"]),
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
            path = self._write_file(
                repo_root, 'def foo():\n    # Use the "simple path" when possible.\n    return 1\n'
            )

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "CACHE_PATH", cache_path),
                patch.object(check_noisy_comments, "_file_line_commit_dates", return_value={}),
            ):
                first = check_noisy_comments.collect_findings(targets=[str(path)], use_cache=True)
                with patch.object(check_noisy_comments, "check_file", side_effect=AssertionError("cache miss")):
                    second = check_noisy_comments.collect_findings(targets=[str(path)], use_cache=True)

            self.assertEqual([finding.code for finding in first], ["NC004"])
            self.assertEqual([finding.code for finding in second], ["NC004"])
            self.assertTrue(cache_path.exists())

    def test_collect_findings_shows_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(
                repo_root, 'def foo():\n    # Use the "simple path" when possible.\n    return 1\n'
            )
            stdout = io.StringIO()

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_file_line_commit_dates", return_value={}),
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
            changed_file.write_text(
                'def foo():\n    # Use the "simple path" when possible.\n    return 1\n', encoding="utf-8"
            )
            unchanged_file = repo_root / "unchanged.py"
            unchanged_file.write_text(
                'def foo():\n    # Use the "second path" when possible.\n    return 1\n', encoding="utf-8"
            )

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(check_noisy_comments, "_changed_python_files_in_patch", return_value={changed_file}),
                patch.object(check_noisy_comments, "_file_line_commit_dates", return_value={}),
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
            small_file.write_text(
                'def foo():\n    # Use the "simple path" when possible.\n    return 1\n', encoding="utf-8"
            )
            stdout = io.StringIO()

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(sys, "argv", ["check_noisy_comments.py", "--path", str(repo_root), "--no-cache"]),
                redirect_stdout(stdout),
            ):
                exit_code = check_noisy_comments.main()

            self.assertEqual(exit_code, 0)
            self.assertLess(stdout.getvalue().index("large.py"), stdout.getvalue().index("small.py"))

    def test_cli_filters_by_rule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(
                repo_root,
                "\n".join(
                    [
                        "def foo():",
                        "    # This is a multi-line note.",
                        "    # It keeps going.",
                        "    # And going.",
                        "    # And going.",
                        "    # And going.",
                        "    # And going.",
                        '    # Use the "simple path" when possible.',
                        "    return 1",
                    ]
                ),
            )
            stdout = io.StringIO()

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(
                    sys, "argv", ["check_noisy_comments.py", "--path", str(path), "--rule", "NC004", "--no-cache"]
                ),
                redirect_stdout(stdout),
            ):
                exit_code = check_noisy_comments.main()

            self.assertEqual(exit_code, 0)
            self.assertIn("NC004", stdout.getvalue())
            self.assertNotIn("NC001", stdout.getvalue())

    def test_cutoff_filter_ignores_old_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(
                repo_root, 'def foo():\n    # Use the "simple path" when possible.\n    return 1\n'
            )
            finding = check_noisy_comments.Finding(
                path=path,
                line=2,
                end_line=2,
                code="NC004",
                message="",
                text="",
                score=1,
            )

            with patch.object(check_noisy_comments, "_file_line_commit_dates", return_value={2: date(2024, 12, 31)}):
                findings = check_noisy_comments._filter_findings_by_cutoff([finding], date(2025, 1, 1))

            self.assertEqual(findings, [])

    def test_cutoff_filter_keeps_findings_on_cutoff_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(
                repo_root, 'def foo():\n    # Use the "simple path" when possible.\n    return 1\n'
            )
            finding = check_noisy_comments.Finding(
                path=path,
                line=2,
                end_line=2,
                code="NC004",
                message="",
                text="",
                score=1,
            )

            with patch.object(check_noisy_comments, "_file_line_commit_dates", return_value={2: date(2025, 1, 1)}):
                findings = check_noisy_comments._filter_findings_by_cutoff([finding], date(2025, 1, 1))

            self.assertEqual(findings, [finding])

    def test_cli_can_fail_on_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            path = self._write_file(
                repo_root, 'def foo():\n    # Use the "simple path" when possible.\n    return 1\n'
            )

            with (
                patch.object(check_noisy_comments, "ROOT", repo_root),
                patch.object(sys, "argv", ["check_noisy_comments.py", str(path), "--fail-on-findings", "--no-cache"]),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = check_noisy_comments.main()

            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
