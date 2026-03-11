#!/bin/bash
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Generation Scheduler — Test Runner (Linux/macOS)
#
# Usage:
#   ./scripts/run_scheduler_tests.sh                    # All quick tests
#   ./scripts/run_scheduler_tests.sh --slow             # Include slow tests
#   ./scripts/run_scheduler_tests.sh --unit-only        # Unit tests only
#   ./scripts/run_scheduler_tests.sh --integration-only # Integration tests only
#   ./scripts/run_scheduler_tests.sh --mode force       # FORCE mode only
#   ./scripts/run_scheduler_tests.sh --verbose          # Verbose output
#   ./scripts/run_scheduler_tests.sh --html-report      # HTML report
#   ./scripts/run_scheduler_tests.sh --help             # Show help

set -euo pipefail

# Navigate to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo ""
echo "========================================================================"
echo "  Generation Scheduler — Test Runner (Bash)"
echo "========================================================================"
echo ""

# Pass all arguments to the Python runner
python scripts/run_scheduler_tests.py "$@"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "  ✓ ALL TESTS PASSED"
else
    echo ""
    echo "  ✗ SOME TESTS FAILED (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
