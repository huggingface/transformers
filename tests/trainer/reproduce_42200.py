"""Run regression tests for issue #42200 with a direct python command.

Examples:
  python tests/trainer/reproduce_42200.py
  python tests/trainer/reproduce_42200.py -k eval_on_start
"""

import argparse

import pytest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run trainer regression tests for issue #42200")
    parser.add_argument("-k", "--keyword", default=None, help="Optional pytest -k expression")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose pytest output")
    args = parser.parse_args()

    test_nodes = [
        "tests/trainer/test_trainer.py::TrainerIntegrationTest::test_compute_metrics_logits_not_tuple_for_causal_lm_eval_on_start",
        "tests/trainer/test_trainer.py::TrainerIntegrationTest::test_compute_metrics_logits_not_tuple_for_causal_lm_predict",
        "tests/trainer/test_trainer.py::TrainerIntegrationTest::test_compute_metrics_with_preprocess_logits_for_metrics_for_causal_lm",
    ]

    pytest_args = []
    if args.verbose:
        pytest_args.append("-vv")
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])
    pytest_args.extend(test_nodes)

    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
