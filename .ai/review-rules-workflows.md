<!-- Read by serge on demand, via the table in /.ai/review-rules.md. Kept out of that file because it is resent on every LLM turn, including serge's model-fix tasks. -->

# Reviewing GitHub Actions workflow changes

A workflow change runs on **every** event in its `on:` block, not just the one the PR is about. Review it once per event.

- **Spell out each event.** For each trigger (`pull_request`, `merge_group`, `push`, `schedule`, …), say what the change does on it. A job added for one event with no `if: github.event_name == '…'` runs on all of them. It adds a runner pickup to the critical path of every run, and if it fails, everything that `needs` it is skipped.
- **Skipped `needs`.** A job whose `needs` was skipped is skipped too, unless its `if:` calls a status function. Use `${{ !cancelled() && needs.X.result != 'failure' }}`: `always()` also runs after a cancellation.
- **Outputs of a skipped or failed job are `''`.** Any expression built from `needs.X.outputs.Y`, and above all a `concurrency.group`, needs a fallback (e.g. `|| format('run-{0}', github.run_id)`). An empty key puts every run into the same group, and with `cancel-in-progress: true` they cancel one another.
- **Concurrency groups: evaluate the string.** Write out the group value for each event. Two kinds of bug: runs that share a group when they should not, and a group meant to dedupe that differs between runs (e.g. `refs/heads/gh-readonly-queue/main/pr-<N>-<base_sha>` changes on every merge-queue rebuild).
- **Required checks.** If a change can skip, rename or fail the job behind a required check (`pr-ci / PR CI status`), say whether the PR becomes unmergeable, or mergeable without the check.
- **Least privilege.** A new job that needs no token gets `permissions: {}` (see `/.github/workflows/get-pr-number.yml`). Never widen workflow-level `permissions`. Pass untrusted context (`github.head_ref`, PR title/body, branch names) through `env:`, never interpolate it into `run:`. `pull_request_target` or `workflow_run` together with a checkout of PR code is a security finding.
- **Runner size.** A job that only runs shell (no checkout, no Docker) belongs on `ubuntu-slim`, not on a full `ubuntu-22.04` runner.
- **GitHub semantics are checkable.** When a conclusion depends on how GitHub Actions behaves (`needs`/`if` evaluation, concurrency scope, reusable-workflow cancellation, which contexts are available where), `fetch_url` the docs.github.com page and cite it, instead of reasoning from memory or telling the author to "trial it".
