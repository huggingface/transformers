from transformers import AutoConfig
from helpers import HF_TOKEN, fetch_latest_ci_results

__all__ = ["check_ci"]


def check_ci(model_id: str) -> str:
    try:
        model_type = AutoConfig.from_pretrained(
            model_id, token=HF_TOKEN, trust_remote_code=True
        ).model_type
    except Exception as e:
        return f"❌ Could not determine model type: `{e}`"

    ci_key = f"models_{model_type.replace('-', '_')}"
    lines = [f"### Daily CI — `{model_type}`"]

    ci_date, ci_data = fetch_latest_ci_results()
    if ci_data is None:
        return "\n".join(lines + ["⚠️ Could not fetch CI results dataset"])

    lines.append(f"_Results from {ci_date}_")

    if ci_key not in ci_data:
        close = [k for k in ci_data if model_type[:5] in k][:5]
        lines.append(f"⚠️ No entry for `{ci_key}`")
        if close:
            lines.append(f"Similar: {', '.join(f'`{k}`' for k in close)}")
        return "\n".join(lines)

    r = ci_data[ci_key]
    total_failed = sum(v for cat in r.get("failed", {}).values() for v in cat.values())
    success, skipped = r.get("success", 0), r.get("skipped", 0)
    has_error = r.get("error", False)

    if has_error:
        lines.append("\n**Status: ❌ job errored**")
    elif total_failed > 0:
        lines.append(f"\n**Status: ❌ {total_failed} test(s) failing**")
    else:
        lines.append(f"\n**Status: ✅ all passing**")

    lines.append(f"✅ {success} passed &nbsp;•&nbsp; ❌ {total_failed} failed &nbsp;•&nbsp; ⏭️ {skipped} skipped")

    failed_cats = {cat: c for cat, c in r.get("failed", {}).items() if any(c.values())}
    if failed_cats:
        lines.append("\n**Failures by category:**")
        for cat, counts in failed_cats.items():
            lines.append(f"- `{cat}` — {', '.join(f'{k}: {v}' for k, v in counts.items() if v)}")

    all_failures = [(rt, item) for rt, items in r.get("failures", {}).items() for item in items]
    if all_failures:
        lines.append(f"\n**Failed tests ({len(all_failures)} total):**")
        for run_type, item in all_failures[:6]:
            parts = item.get("line", "").split("::")
            short = f"{parts[-2]}::{parts[-1]}" if len(parts) >= 3 else item.get("line", "")
            trace = item.get("trace", "").strip()
            lines.append(f"\n`[{run_type}] {short}`")
            if trace:
                preview = trace[:300].replace("\n", " ↵ ")
                lines.append(f"> {preview}{'...' if len(trace) > 300 else ''}")
        if len(all_failures) > 6:
            lines.append(f"\n_...and {len(all_failures) - 6} more_")

    job_links = r.get("job_link", {})
    if job_links:
        lines.append(f"\n**CI job:** {' &nbsp;•&nbsp; '.join(f'[{k}]({v})' for k, v in job_links.items() if v)}")

    return "\n".join(lines)
