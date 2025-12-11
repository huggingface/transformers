#!/usr/bin/env python3
"""
Batch process all models with auto_update.py and collect results.

This script runs auto_update.py for each model in the captured/ directory,
captures the output, and generates a comprehensive report.

Usage:
    python3 batch_update.py [target_directory...] [--apply]

Arguments:
    target_directory  One or more directories to process (default: captured)
                     Can be:
                       - captured (processes all models)
                       - captured/gemma3 (processes just gemma3)
                       - captured/gemma3 captured/florence2 (processes both)
                       - any directory containing captured_info.txt files

Options:
    --apply     Apply changes (default is dry-run)

Examples:
    python3 batch_update.py                           # Process all models in captured/
    python3 batch_update.py captured/gemma3           # Process only gemma3
    python3 batch_update.py captured/gemma3 captured/florence2  # Process gemma3 and florence2
    python3 batch_update.py my_tests/ --apply         # Process my_tests/ and apply changes
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_all_models(base_dir="captured"):
    """Get list of all model directories with captured_info.txt."""
    models = []

    # If base_dir points to a specific model directory with captured_info.txt
    if os.path.isdir(base_dir):
        captured_info = os.path.join(base_dir, "captured_info.txt")
        if os.path.exists(captured_info):
            # This is a single model directory
            return [base_dir]

    # Otherwise, scan the directory for model subdirectories
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} directory not found!")
        return []

    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            captured_info = os.path.join(item_path, "captured_info.txt")
            if os.path.exists(captured_info):
                models.append(item_path)

    return models


def run_update_for_model(model_path, apply=False):
    """
    Run auto_update.py for a single model and capture the result.

    Args:
        model_path: Path to the model directory (can be relative or include 'captured/')
        apply: Whether to apply changes or do dry-run

    Returns:
        dict with: model, success, output, error, exit_code
    """
    # Extract just the model name for display
    model_name = os.path.basename(model_path)
    captured_info_path = os.path.join(model_path, "captured_info.txt")

    cmd = ["python3", "auto_update.py", captured_info_path]
    if apply:
        cmd.append("--apply")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per model
        )

        return {
            "model": model_name,
            "model_path": model_path,
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "model": model_name,
            "model_path": model_path,
            "success": False,
            "output": "",
            "error": "TIMEOUT: Process exceeded 5 minutes",
            "exit_code": -1
        }
    except Exception as e:
        return {
            "model": model_name,
            "model_path": model_path,
            "success": False,
            "output": "",
            "error": f"EXCEPTION: {str(e)}",
            "exit_code": -2
        }


def extract_summary(output):
    """Extract key information from the output."""
    lines = output.split('\n')
    summary = {
        "blocks": 0,
        "skipped": 0,
        "tasks_found": 0,
        "files_updated": 0,
        "completed": False,
        "update_failures": 0,
        "analysis_failures": 0,
        "has_failures": False
    }

    for line in lines:
        # Extract statistics from STATS line
        if line.startswith("STATS:"):
            parts = line.split()
            for part in parts[1:]:  # Skip "STATS:"
                if "=" in part:
                    key, value = part.split("=")
                    if key == "blocks":
                        summary["blocks"] = int(value)
                    elif key == "skip":
                        summary["skipped"] = int(value)
                    elif key == "tasks":
                        summary["tasks_found"] = int(value)
        if "Found" in line and "update task(s)" in line:
            try:
                summary["tasks_found"] = int(line.split("Found")[1].split("update")[0].strip())
            except:
                pass
        if "Grouped into" in line and "file(s)" in line:
            try:
                summary["files_updated"] = int(line.split("Grouped into")[1].split("file")[0].strip())
            except:
                pass
        if "✅ Complete!" in line or "Complete!" in line:
            summary["completed"] = True
        # Check for failures
        if "✗ Update failed" in line:
            summary["update_failures"] += 1
            summary["has_failures"] = True
        if "✗ Could not analyze" in line:
            summary["analysis_failures"] += 1
            summary["has_failures"] = True
        if "✗ Unknown pattern" in line:
            summary["update_failures"] += 1
            summary["has_failures"] = True

    return summary


def main():
    apply = "--apply" in sys.argv

    # Get target directories from command line args
    target_dirs = []
    for arg in sys.argv[1:]:
        if arg != "--apply" and not arg.startswith("-"):
            target_dirs.append(arg)

    # Default to "captured" if no directories specified
    if not target_dirs:
        target_dirs = ["captured"]

    mode = "APPLY MODE" if apply else "DRY RUN"

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"batch_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Open combined output file
    combined_output_file = os.path.join(results_dir, "ALL_OUTPUT.txt")
    combined_f = open(combined_output_file, 'w')

    print("=" * 80)
    print(f"Batch Update Script - {mode}")
    print("=" * 80)
    print()

    # Get all models from all target directories
    all_models = []
    for target_dir in target_dirs:
        models = get_all_models(target_dir)
        all_models.extend(models)

    if not all_models:
        print(f"No models found in: {', '.join(target_dirs)}")
        combined_f.close()
        return

    print(f"Found {len(all_models)} models to process")
    print(f"Target directories: {', '.join(target_dirs)}")
    print(f"Results will be saved to: {results_dir}/")
    print(f"Combined output: {combined_output_file}")
    print()

    # Process each model
    results = []
    success_count = 0
    failure_count = 0

    for i, model_path in enumerate(all_models, 1):
        model_name = os.path.basename(model_path)
        print(f"[{i}/{len(all_models)}] Processing {model_name}...", end=" ", flush=True)

        # Write separator to combined output
        captured_info_path = os.path.join(model_path, "captured_info.txt")
        separator = "=" * 80
        combined_f.write(f"\n{separator}\n")
        combined_f.write(f"Running with {captured_info_path}\n")
        combined_f.write(f"{separator}\n\n")
        combined_f.flush()

        result = run_update_for_model(model_path, apply)
        results.append(result)

        # Write output to combined file
        combined_f.write(result['output'])
        if result['error']:
            combined_f.write(f"\nSTDERR:\n{result['error']}\n")
        combined_f.write("\n")
        combined_f.flush()

        # Save individual result
        output_file = os.path.join(results_dir, f"{result['model']}.txt")
        with open(output_file, 'w') as f:
            f.write(f"Model: {result['model']}\n")
            f.write(f"Status: {'SUCCESS' if result['success'] else 'FAILED'}\n")
            f.write(f"Exit Code: {result['exit_code']}\n")
            f.write("=" * 80 + "\n")
            f.write("STDOUT:\n")
            f.write(result['output'])
            f.write("\n" + "=" * 80 + "\n")
            if result['error']:
                f.write("STDERR:\n")
                f.write(result['error'])

        # Extract summary and check for failures
        summary = extract_summary(result['output'])

        # Mark as failed if there were update failures even if exit code is 0
        if summary['has_failures']:
            result['has_update_failures'] = True
            result['update_failure_count'] = summary['update_failures'] + summary['analysis_failures']
        else:
            result['has_update_failures'] = False
            result['update_failure_count'] = 0

        # Print status
        if result['success'] and not summary['has_failures']:
            success_count += 1
            status_parts = []
            if summary['blocks'] > 0:
                status_parts.append(f"blocks: {summary['blocks']}")
            if summary['skipped'] > 0:
                status_parts.append(f"skip: {summary['skipped']}")
            status_parts.append(f"tasks: {summary['tasks_found']}")
            status_str = ", ".join(status_parts)
            print(f"✅ OK ({status_str})")
        elif result['success'] and summary['has_failures']:
            failure_count += 1
            status_parts = []
            if summary['blocks'] > 0:
                status_parts.append(f"blocks: {summary['blocks']}")
            if summary['skipped'] > 0:
                status_parts.append(f"skip: {summary['skipped']}")
            status_parts.append(f"tasks: {summary['tasks_found']}")
            updated = summary['tasks_found'] - summary['update_failures'] - summary['analysis_failures']
            status_parts.append(f"updated: {updated}")
            total_failures = summary['update_failures'] + summary['analysis_failures']
            status_parts.append(f"failures: {total_failures}")
            status_str = ", ".join(status_parts)
            print(f"⚠️  PARTIAL ({status_str})")
        else:
            failure_count += 1
            print(f"❌ FAILED (exit code: {result['exit_code']})")

    print()
    print("=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print()

    # Close combined output file
    combined_f.close()

    # Generate summary report
    summary_file = os.path.join(results_dir, "SUMMARY.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Batch Update Summary - {mode}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Models: {len(all_models)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {failure_count}\n")
        f.write("=" * 80 + "\n\n")

        # List all results
        f.write("Detailed Results:\n")
        f.write("-" * 80 + "\n")
        for result in results:
            summary = extract_summary(result['output'])
            if result['success'] and not summary['has_failures']:
                status = "✅ SUCCESS"
            elif result['success'] and summary['has_failures']:
                status = f"⚠️  PARTIAL ({summary['update_failures']}U+{summary['analysis_failures']}A fails)"
            else:
                status = "❌ FAILED"
            f.write(
                f"{result['model']:<30} {status:<25} Tasks: {summary['tasks_found']:<3} Files: {summary['files_updated']:<3}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Models with Issues:\n")
        f.write("-" * 80 + "\n")
        failed_models = [r for r in results if not r['success'] or r.get('has_update_failures', False)]
        if failed_models:
            for result in failed_models:
                f.write(f"\n{result['model']}:\n")
                if result.get('has_update_failures', False):
                    f.write(f"  Update failures: {result['update_failure_count']}\n")
                if result['error']:
                    f.write(f"  Error: {result['error'][:200]}\n")
                elif not result['success']:
                    f.write(f"  Exit code: {result['exit_code']}\n")
        else:
            f.write("None! All models processed successfully.\n")

    # Print summary to console
    print(f"Summary:")
    print(f"  Total:      {len(all_models)}")
    print(f"  Successful: {success_count} ({100 * success_count // len(all_models)}%)")
    print(f"  Failed:     {failure_count} ({100 * failure_count // len(all_models)}%)")
    print()
    print(f"Results saved to: {results_dir}/")
    print(f"  - Combined output: {results_dir}/ALL_OUTPUT.txt")
    print(f"  - Individual logs: {results_dir}/<model_name>.txt")
    print(f"  - Summary report:  {results_dir}/SUMMARY.txt")
    print()

    # Show failed models if any
    if failure_count > 0:
        print("Models with issues:")
        for result in results:
            if not result['success']:
                print(f"  - {result['model']} (exit code: {result['exit_code']})")
            elif result.get('has_update_failures', False):
                print(f"  - {result['model']} (partial: {result['update_failure_count']} update failures)")
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()