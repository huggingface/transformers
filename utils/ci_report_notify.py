"""
Upload the CI failure report files of a workflow run (written by analyze_ci_report.py) to a Hub dataset repo,
then post a summary of the run to Slack.

Hub layout (like the daily CI's `hf-internal-testing/transformers_daily_ci`):
  <YYYY-MM-DD>/<file>.json                                  scheduled runs (date = the run's creation date)
  <YYYY-MM-DD>/runs/<run_number>-<run_id>/<file>.json       other runs (push, workflow_dispatch)
File names are those of the reports without their date suffix (e.g. `failing_tests.json`).

Usage:
  python utils/ci_report_notify.py --reports-dir <DIR> --hub-repo <ORG/NAME> --slack-channel <#channel> \
      [--commit <NAME>=<SHA> ...] [--commit-date <NAME>=<DATE> ...] [--dry-run]

Below its title, the Slack message shows the transformers commit of the run, followed by one line per
--commit (e.g. `--commit vllm=<SHA>` for the vLLM commit that was tested). A commit's date, given with
--commit-date (e.g. `--commit-date transformers=<DATE>`), is shown after it.

Environment:
  TRANSFORMERS_CI_RESULTS_UPLOAD_TOKEN   Hub token with write access to --hub-repo
  CI_SLACK_BOT_TOKEN                     Slack bot token (chat:write) for --slack-channel

With --dry-run, nothing is uploaded or posted: the Slack payload is printed instead.
"""

import argparse
import glob
import json
import os
import re
import sys
import urllib.request


# Slack limits: 3000 characters per section text, 50 blocks per message.
_MAX_SECTION_CHARS = 2900
_MAX_LISTED = 10


def _load(reports_dir, name):
    """Load a report file by its name without the date suffix, e.g. "new_failing_tests"."""
    paths = glob.glob(os.path.join(reports_dir, f"{name}.json")) + glob.glob(
        os.path.join(reports_dir, f"{name}_[0-9][0-9][0-9][0-9]_[0-9][0-9]_[0-9][0-9].json")
    )
    if not paths:
        return None
    with open(paths[0], encoding="utf-8") as f:
        return json.load(f)


def hub_folder(run):
    date = run["created_at"].split("T")[0]
    if run["event"] == "schedule":
        return date
    return f"{date}/runs/{run['number']}-{run['id']}"


def upload_reports(reports_dir, hub_repo, folder, token):
    """Upload all report files in one commit and return the commit sha."""
    from huggingface_hub import CommitOperationAdd, HfApi

    operations = []
    for path in sorted(glob.glob(os.path.join(reports_dir, "*.json"))):
        name = re.sub(r"_\d{4}_\d{2}_\d{2}\.json$", ".json", os.path.basename(path))
        operations.append(CommitOperationAdd(path_in_repo=f"{folder}/{name}", path_or_fileobj=path))
    commit = HfApi().create_commit(
        repo_id=hub_repo,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Upload CI reports to {folder}",
        token=token,
    )
    return commit.oid


def _escape(text):
    """Escape text from the logs for Slack mrkdwn, so e.g. "<!channel>" or "<url|text>" in an error is shown as is."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _section(text):
    if len(text) > _MAX_SECTION_CHARS:
        text = text[:_MAX_SECTION_CHARS] + "\n…"
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _button(text, url):
    return {"type": "button", "text": {"type": "plain_text", "text": text}, "url": url}


def build_slack_blocks(job_results, new_failing, categories, hub_url, new_failures_url, commits=(), commit_dates=None):
    """
    `commits`: (name, sha) pairs shown after the transformers commit, e.g. [("vllm", "<SHA>")].
    `commit_dates`: name -> date shown after the commit of that name, e.g. {"transformers": "<DATE>"}.
    """
    commit_dates = commit_dates or {}
    run = job_results["run"]
    jobs = job_results["jobs"]

    title = run["workflow"] if run["event"] == "schedule" else f"{run['workflow']} ({run['event']} on {run['branch']})"
    totals = {
        key: sum(job.get(key) or 0 for job in jobs.values()) for key in ("passed", "failed", "errors", "skipped")
    }
    num_failed = totals["failed"] + totals["errors"]
    num_new = new_failing["total_new_failing"] if new_failing else 0
    no_summary = {name: job for name, job in jobs.items() if job["status"] == "no_summary"}

    if num_failed == 0 and not no_summary:
        status = f"✅ All tests passed: {totals['passed']} passed, {totals['skipped']} skipped."
    else:
        status = (
            f"❌ {num_failed} failed tests ({num_new} new), {totals['passed']} passed, {totals['skipped']} skipped."
        )

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": title[:150]}},
        _section(
            "\n".join(
                f"{name} commit: {sha or '(unknown)'}" + (f" ({commit_dates[name]})" if commit_dates.get(name) else "")
                for name, sha in [("transformers", run["head_sha"]), *commits]
            )
        ),
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": status},
            "accessory": _button("Check Action results", run["url"]),
        },
    ]

    if no_summary:
        jobs_text = "1 job" if len(no_summary) == 1 else f"{len(no_summary)} jobs"
        lines = [f"⚠️ {jobs_text} ended without a pytest summary (killed, setup failure or log unavailable):"]
        lines += [
            f"• <{job['job_link']}|{_escape(name)}>: {_escape(job['error'][:150])}" for name, job in no_summary.items()
        ]
        blocks.append(_section("\n".join(lines)))

    failed_jobs = {name: job for name, job in jobs.items() if job.get("failed") or job.get("errors")}
    if failed_jobs:
        lines = ["*Failed jobs* (failed tests, new):"]
        lines += [
            f"• <{job['job_link']}|{_escape(name)}>: {job['failed'] + job['errors']} ({job['new_failures']} new)"
            for name, job in failed_jobs.items()
        ]
        blocks.append(_section("\n".join(lines)))

    if num_new:
        new_tests = [test for job in new_failing["jobs"].values() for test in job["tests"]]
        listed = _escape("\n".join(new_tests[:_MAX_LISTED]))
        more = f"\n… and {len(new_tests) - _MAX_LISTED} more" if len(new_tests) > _MAX_LISTED else ""
        blocks.append(_section(f"*New failing tests:*\n```\n{listed}\n```{more}"))

    if categories and categories["categories"]:
        top = sorted(categories["categories"].items(), key=lambda kv: -kv[1]["count"])[:5]
        lines = ["*Top errors:*"] + [f"• ({cat['count']}) `{_escape(error[:200])}`" for error, cat in top]
        blocks.append(_section("\n".join(lines)))

    buttons = [_button("New failures (Hub)", new_failures_url)] if num_new else []
    buttons.append(_button("All results (Hub)", hub_url))
    blocks.append({"type": "actions", "elements": buttons})

    return title, blocks


def post_to_slack(channel, text, blocks, token):
    request = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps({"channel": channel, "text": text, "blocks": blocks}).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(request, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    # Slack answers HTTP 200 with {"ok": false, "error": ...} on errors (e.g. "not_in_channel").
    if not result.get("ok"):
        raise RuntimeError(f"Slack API error: {result.get('error')}")


def main():
    parser = argparse.ArgumentParser(description="Upload CI reports to the Hub and post a summary to Slack.")
    parser.add_argument(
        "--reports-dir", required=True, help="Directory with the files written by analyze_ci_report.py"
    )
    parser.add_argument(
        "--hub-repo", required=True, help="Hub dataset repo, e.g. hf-internal-testing/transformers_vllm_ci"
    )
    parser.add_argument("--slack-channel", required=True, help="Slack channel, e.g. #transformers-ci-vllm")
    parser.add_argument(
        "--commit",
        action="append",
        default=[],
        metavar="NAME=SHA",
        help="Commit of another tested project to show in the Slack message, e.g. vllm=<SHA>. Can be repeated.",
    )
    parser.add_argument(
        "--commit-date",
        action="append",
        default=[],
        metavar="NAME=DATE",
        help="Date shown after the commit of NAME (`transformers` or a --commit NAME). Can be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't upload or post; print the Slack payload")
    args = parser.parse_args()
    commits = []
    for value in args.commit:
        name, sep, sha = value.partition("=")
        if not sep or not name:
            parser.error(f"--commit must be NAME=SHA, got {value!r}")
        commits.append((name, sha))
    commit_dates = {}
    for value in args.commit_date:
        name, sep, date = value.partition("=")
        if not sep or not name:
            parser.error(f"--commit-date must be NAME=DATE, got {value!r}")
        commit_dates[name] = date

    job_results = _load(args.reports_dir, "job_results")
    if job_results is None:
        sys.exit(f"No job_results.json in {args.reports_dir}: the report was not generated.")
    new_failing = _load(args.reports_dir, "new_failing_tests")
    categories = _load(args.reports_dir, "failing_tests_categories")

    folder = hub_folder(job_results["run"])
    if args.dry_run:
        revision = "main"
    else:
        revision = upload_reports(
            args.reports_dir, args.hub_repo, folder, os.environ["TRANSFORMERS_CI_RESULTS_UPLOAD_TOKEN"]
        )
        print(f"Uploaded reports to https://huggingface.co/datasets/{args.hub_repo}/tree/{revision}/{folder}")
    hub_url = f"https://huggingface.co/datasets/{args.hub_repo}/tree/{revision}/{folder}"
    new_failures_url = (
        f"https://huggingface.co/datasets/{args.hub_repo}/blob/{revision}/{folder}/new_failing_tests_with_trace.json"
    )

    title, blocks = build_slack_blocks(
        job_results, new_failing, categories, hub_url, new_failures_url, commits, commit_dates
    )
    if args.dry_run:
        print(
            json.dumps({"channel": args.slack_channel, "text": title, "blocks": blocks}, indent=2, ensure_ascii=False)
        )
        return
    post_to_slack(args.slack_channel, title, blocks, os.environ["CI_SLACK_BOT_TOKEN"])
    print(f"Posted to {args.slack_channel}")


if __name__ == "__main__":
    main()
