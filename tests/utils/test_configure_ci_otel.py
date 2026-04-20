import importlib.util
import json
import tempfile
from pathlib import Path
from unittest import TestCase


MODULE_PATH = Path(__file__).resolve().parents[2] / "utils" / "configure_ci_otel.py"
SPEC = importlib.util.spec_from_file_location("configure_ci_otel", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
configure_ci_otel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(configure_ci_otel)


class ConfigureCiOtelTests(TestCase):
    def test_prepare_environment_skips_export_without_endpoint(self):
        env, export_traces = configure_ci_otel.prepare_environment({})

        self.assertFalse(export_traces)
        self.assertNotIn("OTEL_SERVICE_NAME", env)
        self.assertNotIn("OTEL_RESOURCE_ATTRIBUTES", env)

    def test_prepare_environment_adds_github_attributes(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as event_file:
            json.dump({"pull_request": {"number": 4321}}, event_file)
            event_file.flush()

            env, export_traces = configure_ci_otel.prepare_environment(
                {
                    "GITHUB_ACTIONS": "true",
                    "GITHUB_EVENT_PATH": event_file.name,
                    "GITHUB_JOB": "run_models_gpu",
                    "GITHUB_REF_NAME": "otel-support",
                    "GITHUB_RUN_ATTEMPT": "2",
                    "GITHUB_RUN_ID": "12345",
                    "GITHUB_SHA": "deadbeef",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
                },
                job_name="models_gpu_slice",
            )

        self.assertTrue(export_traces)
        self.assertEqual(env["OTEL_SERVICE_NAME"], "transformers-tests")
        self.assertIn("transformers.test.provider=github_actions", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn("transformers.test.job=models_gpu_slice", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn("cicd.pipeline.run.id=12345", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn("transformers.test.job.id=12345:models_gpu_slice:2", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn("vcs.change.id=4321", env["OTEL_RESOURCE_ATTRIBUTES"])

    def test_prepare_environment_adds_circleci_attributes(self):
        env, export_traces = configure_ci_otel.prepare_environment(
            {
                "CIRCLE_BRANCH": "pull/987",
                "CIRCLE_BUILD_NUM": "24680",
                "CIRCLE_JOB": "tests_torch",
                "CIRCLE_SHA1": "cafebabe",
                "CIRCLE_WORKFLOW_ID": "workflow-123",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
            },
            job_name="tests_torch",
        )

        self.assertTrue(export_traces)
        self.assertIn("transformers.test.provider=circleci", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn("transformers.test.job=tests_torch", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn("transformers.test.job.id=24680", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn("vcs.change.id=987", env["OTEL_RESOURCE_ATTRIBUTES"])

    def test_prepare_environment_supports_local_forced_export(self):
        env, export_traces = configure_ci_otel.prepare_environment(
            {},
            job_name="local_smoke",
            force_export_traces=True,
        )

        self.assertTrue(export_traces)
        self.assertEqual(env["OTEL_SERVICE_NAME"], "transformers-tests")
        self.assertIn("deployment.environment=local", env["OTEL_RESOURCE_ATTRIBUTES"])
        self.assertIn("transformers.test.job=local_smoke", env["OTEL_RESOURCE_ATTRIBUTES"])

    def test_augment_pytest_command_adds_export_flag_once(self):
        command = ["python3", "-m", "pytest", "tests/utils"]

        augmented_command = configure_ci_otel.augment_pytest_command(command, export_traces=True)
        augmented_again = configure_ci_otel.augment_pytest_command(
            augmented_command,
            export_traces=True,
        )

        self.assertEqual(augmented_command[-1], "--export-traces")
        self.assertEqual(augmented_again.count("--export-traces"), 1)

    def test_configure_trace_context_generates_traceparent_for_pytest(self):
        env, trace_id = configure_ci_otel.configure_trace_context(
            {},
            ["python3", "-m", "pytest", "tests/utils"],
            export_traces=True,
        )

        self.assertIsNotNone(trace_id)
        self.assertIn("TRACEPARENT", env)
        self.assertEqual(trace_id, configure_ci_otel.trace_id_from_traceparent(env["TRACEPARENT"]))

    def test_configure_trace_context_preserves_existing_traceparent(self):
        traceparent = "00-1234567890abcdef1234567890abcdef-fedcba0987654321-01"

        env, trace_id = configure_ci_otel.configure_trace_context(
            {"TRACEPARENT": traceparent},
            ["python3", "-m", "pytest", "tests/utils"],
            export_traces=True,
        )

        self.assertEqual(env["TRACEPARENT"], traceparent)
        self.assertEqual(trace_id, "1234567890abcdef1234567890abcdef")
