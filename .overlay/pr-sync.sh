#!/usr/bin/env bash
# USAGE
#
# pr-sync: rebuild and validate the upstream PR head for the Apertus 1.5 and WavTokenizer
# work (huggingface/transformers#47662).
#
# THE IDEA
#   Two branches are kept in step:
#     feature/apertus_1p5_pipeline   dev, pushed to `origin`
#     add-apertus1p5                 the PR head, pushed to `swissai`
#   They differ by exactly the purely additive "overlay" files listed in
#   .overlay/manifest.txt: local harnesses, parity scripts and conversion tests that must
#   never reach upstream. The PR head is therefore a pure function of dev, PR = dev minus
#   overlay, and is always rebuilt rather than merged or cherry picked.
#   Run from anywhere inside the repository. With no arguments, or -h, prints this text.
#
# COMMANDS
#   Read only:
#     check-host      Preflight: host class (login or compute), interpreter policy, that
#                     transformers imports from this checkout, test extras, how many model
#                     cards add_dates would walk, and which tiers can run here.
#                     Nonzero if blocked.
#     status          Both branches, both remotes, upstream drift, overlay file count.
#     verify-strip    The four invariants, about a second, no strip and no tests:
#                       1 the PR head is dev plus exactly one strip commit
#                       2 every difference between the branches is an addition
#                       3 the differing set equals the manifest expansion exactly
#                       4 no shipped file references a stripped path
#
#   Mutating:
#     fetch-remotes   Fetch all three remotes, fast forward local `main`, record the push
#                     leases. Run this first: it is what keeps add_dates cheap.
#     rebase-dev      Rebase dev onto upstream/main, aborting on conflict.
#     autofix-dev     Run `make fix-repo` on dev and commit the regeneration.
#     strip-to-pr     Strip the manifest paths in a detached side worktree, commit the
#                     strip, move the PR branch ref, then run verify-strip.
#     run-tests       Run the test tiers. Accepts --only=TIER.
#     push            `push --dev` or `push --pr`. Forces only against the lease recorded
#                     by fetch-remotes, and refuses if the remote moved since then.
#     full-cycle      check-host, fetch-remotes, rebase-dev, autofix-dev, strip-to-pr
#                     (which verifies), run-tests. Takes no flags and never pushes.
#
# TIERS
#   checks    `make check-repo` in the PR worktree. What upstream CI runs.
#   fast-pr   pytest inside the PR worktree, over the two model test dirs.
#   fast      the same pytest in the dev tree.
#   slow      RUN_SLOW=1, integration classes only. Dev lane.
#   ap        the ap_testcase scripts, dev only. 01 and 02 build the processor alone;
#             03 onward load the full 8B model on CPU; 08 needs two GPUs, 09 needs one.
#
# ENVIRONMENT
#   PRSYNC_PYTHON     override the interpreter check-host would pick for this host.
#   PRSYNC_MANIFEST   alternate manifest, used to exercise the invariants in testing.
#
# TYPICAL WORKFLOW
#   After an upstream rebase, from the dev branch:
#
#     .overlay/pr-sync.sh full-cycle    # check-host runs first, inside full-cycle
#     .overlay/pr-sync.sh push --dev
#     .overlay/pr-sync.sh push --pr
#
#   Quick loop while iterating, skipping the expensive tiers:
#
#     .overlay/pr-sync.sh autofix-dev && .overlay/pr-sync.sh strip-to-pr
#     .overlay/pr-sync.sh run-tests --only=fast
#
#   Prove the branches are still consistent, about a second:
#
#     .overlay/pr-sync.sh verify-strip
#
#   For the full slow suite including the inherited mixin tests, run pytest directly:
#     RUN_SLOW=1 python -m pytest tests/models/apertus1p5 tests/models/wavtokenizer -n auto
#
# END USAGE

set -euo pipefail
shopt -s inherit_errexit 2>/dev/null || true

DEV_BRANCH=feature/apertus_1p5_pipeline
PR_BRANCH=add-apertus1p5
PR_REMOTE=swissai
DEV_REMOTE=origin
UPSTREAM=upstream

CONDA_PY=/users/rkreft/miniconda3/envs/myenv/bin/python
GLOBAL_PY=/usr/bin/python
STATE_DIR=$HOME/.cache/pr-sync
TEST_PATHS=(tests/models/apertus1p5 tests/models/wavtokenizer)
TIERS=(checks fast-pr fast slow ap)

ONLY_TIER=""

# ------------------------------------------------------------------------------- ui

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_RST=$'\033[0m'; C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'
else
    C_RST=''; C_R=''; C_G=''; C_Y=''
fi

log()  { printf '==> %s\n' "$*"; }
ok()   { printf '  %sok%s   %s\n' "$C_G" "$C_RST" "$*"; }
warn() { printf '  %swarn%s %s\n' "$C_Y" "$C_RST" "$*" >&2; }
bad()  { printf '  %sfail%s %s\n' "$C_R" "$C_RST" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$C_R" "$C_RST" "$*" >&2; exit 1; }
note() { printf '  %s\n' "$*"; }

# --------------------------------------------------------------------------- helpers

# Resolved before the cd, so --help still works when invoked from a subdirectory.
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || die "not inside a git repository"
cd "$ROOT"

# PRSYNC_MANIFEST is how the invariants get exercised against a synthetic manifest without
# committing to dev. Delete it, and the conditional below, once .overlay is committed.
MANIFEST=${PRSYNC_MANIFEST:-$ROOT/.overlay/manifest.txt}
WORKTREE=$(dirname "$ROOT")/pr-wt
mkdir -p "$STATE_DIR"

SPECS=()
load_manifest() {
    [ -f "$MANIFEST" ] || die "missing manifest: $MANIFEST"
    mapfile -t SPECS < <(sed -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "$MANIFEST" | grep -v '^$')
    [ ${#SPECS[@]} -gt 0 ] || die "manifest is empty"
    local spec
    for spec in "${SPECS[@]}"; do
        case "$spec" in
            /*|*..*|*'*'*|*'?'*|*'['*|:*)
                die "manifest: unsupported pathspec '$spec'. Literal relative paths only: git ls-tree ignores globs silently, so a glob would expand to nothing and the file would ship." ;;
        esac
    done
    if [ -z "${PRSYNC_MANIFEST:-}" ]; then
        printf '%s\n' "${SPECS[@]}" | grep -qxF '.overlay' \
            || die "manifest must list '.overlay' so the tooling strips with the rest of the overlay"
    fi
}

# Every entry must match at least one tracked file: git ls-tree exits 0 on a no-match, so
# without this a typo expands to nothing and the invariant passes while the file ships.
expand_manifest() {
    local rev=$1 spec out
    for spec in "${SPECS[@]}"; do
        out=$(git ls-tree -r --name-only "$rev" -- "$spec")
        [ -n "$out" ] || die "manifest entry matches nothing at $rev: $spec"
        printf '%s\n' "$out"
    done
}

current_branch() { git branch --show-current; }
rev() { git rev-parse --verify -q "$1" 2>/dev/null || true; }

assert_branch() {
    local have; have=$(current_branch)
    [ "$have" = "$1" ] || die "wrong branch: on '$have', expected '$1'"
}

assert_clean() {
    [ -z "$(git status --porcelain --untracked-files=no)" ] \
        || die "working tree is dirty; commit or stash first"
}

# add_dates scopes its work with `git merge-base main HEAD`, against the LOCAL main.
# A stale main makes every check-repo walk hundreds of unrelated model cards.
card_count() { git diff --name-only "$(git merge-base main HEAD)" | grep -c 'model_doc/.*\.md' || true; }

# --------------------------------------------------------------------- host detection

HOST_CLASS=""; GPUS=0; PY=""

detect_host() {
    GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
    if [ "$GPUS" -eq 0 ]; then HOST_CLASS=login; PY=$CONDA_PY
    elif [ "$GPUS" -eq 1 ]; then HOST_CLASS=compute-1gpu; PY=$GLOBAL_PY
    else HOST_CLASS=compute-multi; PY=$GLOBAL_PY
    fi
    [ -n "${PRSYNC_PYTHON:-}" ] && PY=$PRSYNC_PYTHON
    [ -x "$PY" ] || die "interpreter not executable: $PY"
}

# Echoes why the tier cannot run here, or nothing if it can. The trailing `return 0` is
# load bearing: without it the function exits with the status of the last test, which
# errexit would turn into a silent script exit at the calling assignment.
tier_blocker() {
    case "$1" in slow|ap) ;; *) return 0 ;; esac
    [ "$HOST_CLASS" = login ] && { echo "login node"; return 0; }
    local free_gb
    # `|| true` is required: 2>/dev/null hides the message, not a 127 from a missing `free`.
    free_gb=$(free -g 2>/dev/null | awk 'NR==2{print $7}' || true)
    [ -n "$free_gb" ] && [ "$free_gb" -lt 24 ] \
        && echo "only ${free_gb}GB RAM free, need 24 (the 8B model loads on CPU)"
    return 0
}

# ---------------------------------------------------------------------------- doctor

cmd_check_host() {
    [ $# -eq 0 ] || die "check-host takes no flags"
    load_manifest
    detect_host
    local hard=0

    log "host"
    note "class      $HOST_CLASS, $GPUS GPU(s)"
    note "python     $PY"
    if [ "$HOST_CLASS" = login ] && [ "$PY" = "$GLOBAL_PY" ]; then
        bad "global python on a login node (policy requires the conda env)"; hard=1
    else
        ok "interpreter matches host policy"
    fi

    log "environment"
    local tpath
    tpath=$("$PY" -c 'import transformers,sys; sys.stdout.write(transformers.__file__)' 2>/dev/null || true)
    if [ -z "$tpath" ]; then
        bad "cannot import transformers with $PY"; hard=1
    elif [ "${tpath#"$ROOT"/src/}" = "$tpath" ]; then
        bad "transformers resolves to $tpath, not $ROOT/src (ap_testcase requires the local checkout)"; hard=1
    else
        ok "transformers resolves to the local checkout"
    fi
    local missing="" m
    for m in torch torchvision librosa soundfile PIL datasets pytest xdist; do
        "$PY" -c "import $m" >/dev/null 2>&1 || missing="$missing $m"
    done
    [ -n "$missing" ] && warn "missing modules:$missing" || ok "test extras present"

    log "checker cost"
    local cards; cards=$(card_count)
    if [ "$cards" -gt 20 ]; then
        warn "local 'main' is stale: add_dates would walk $cards model cards; 'pr-sync fetch-remotes' cuts this to ~2"
    else
        ok "add_dates scope: $cards model card(s)"
    fi

    log "tiers"
    local t why
    for t in "${TIERS[@]}"; do
        why=$(tier_blocker "$t")
        [ -z "$why" ] && ok "$t" || note "skip $t: $why"
    done

    [ "$hard" = 0 ] || die "check-host found blocking problems"
    log "check-host: ok"
}

# ----------------------------------------------------------------------------- status

cmd_status() {
    [ $# -eq 0 ] || die "status takes no flags"
    local dev pr up r
    dev=$(rev "$DEV_BRANCH"); pr=$(rev "$PR_BRANCH"); up=$(rev "$UPSTREAM/main")
    log "branches"
    note "dev        ${dev:0:10}  $DEV_BRANCH"
    note "pr         ${pr:0:10}  $PR_BRANCH"
    note "upstream   ${up:0:10}  $UPSTREAM/main"
    log "remotes"
    r=$(rev "$DEV_REMOTE/$DEV_BRANCH")
    [ -n "$r" ] && note "$DEV_REMOTE     ${r:0:10}  $([ "$r" = "$dev" ] && echo 'in sync' || echo DIFFERS)"
    r=$(rev "$PR_REMOTE/$PR_BRANCH")
    [ -n "$r" ] && note "$PR_REMOTE    ${r:0:10}  $([ "$r" = "$pr" ] && echo 'in sync' || echo DIFFERS)"
    if [ -n "$pr" ] && [ -n "$up" ]; then
        log "pr vs upstream"
        note "$(git rev-list --count "$pr..$up") behind, $(git rev-list --count "$up..$pr") ahead"
    fi
    if [ -n "$dev" ] && [ -n "$pr" ]; then
        note "overlay: $(git diff --name-only "$pr" "$dev" | wc -l) file(s) differ"
    fi
    return 0
}

# ------------------------------------------------------------------------------- sync

cmd_fetch_remotes() {
    [ $# -eq 0 ] || die "fetch-remotes takes no flags"
    log "fetching"
    git fetch -q "$UPSTREAM" main
    git fetch -q "$DEV_REMOTE" "$DEV_BRANCH" || warn "could not fetch $DEV_REMOTE/$DEV_BRANCH"
    git fetch -q "$PR_REMOTE" "$PR_BRANCH" || warn "could not fetch $PR_REMOTE/$PR_BRANCH"

    # Leases are captured now, at cycle start. Re-fetching immediately before a push would
    # lease against whatever someone else just pushed, which defeats the protection.
    rev "$PR_REMOTE/$PR_BRANCH"   > "$STATE_DIR/lease_pr"
    rev "$DEV_REMOTE/$DEV_BRANCH" > "$STATE_DIR/lease_dev"
    ok "leases recorded"

    if [ "$(current_branch)" = main ]; then
        warn "on 'main'; not moving it (that would leave a phantom diff in the tree)"
    elif git merge-base --is-ancestor main "$UPSTREAM/main" 2>/dev/null; then
        local before; before=$(card_count)
        git update-ref refs/heads/main "$(git rev-parse "$UPSTREAM/main")"
        ok "local main fast-forwarded; add_dates scope $before -> $(card_count) cards"
    else
        warn "local main has diverged from $UPSTREAM/main; add_dates will walk $(card_count) cards"
    fi
}

# ----------------------------------------------------------------------------- rebase

cmd_rebase_dev() {
    [ $# -eq 0 ] || die "rebase-dev takes no flags"
    assert_branch "$DEV_BRANCH"; assert_clean
    local before after
    before=$(git rev-parse HEAD)
    log "rebasing $DEV_BRANCH onto $UPSTREAM/main"
    git rebase "$UPSTREAM/main" || { git rebase --abort || true; die "rebase conflicted and was aborted"; }
    after=$(git rev-parse HEAD)
    [ "$before" = "$after" ] && ok "already up to date" || ok "rebased ${before:0:10} -> ${after:0:10}"

    # The editable install is a bare path append, so a branch flip needs no reinstall.
    # Only a packaging change does.
    local h prev
    h=$(cat setup.py pyproject.toml src/transformers/dependency_versions_table.py 2>/dev/null | sha1sum | cut -d' ' -f1) || h=""
    prev=$(cat "$STATE_DIR/install.hash" 2>/dev/null || true)
    if [ -n "$h" ] && [ "$h" != "$prev" ]; then
        warn "packaging files changed; reinstall then record:"
        note "  \$PY -m pip install -e '.[testing,vision,audio]' && echo $h > $STATE_DIR/install.hash"
    else
        ok "packaging unchanged, no reinstall needed"
    fi
}

# -------------------------------------------------------------------------- normalize

cmd_autofix_dev() {
    [ $# -eq 0 ] || die "autofix-dev takes no flags"
    assert_branch "$DEV_BRANCH"
    # Untracked files must be included here, unlike assert_clean: the commit below stages
    # with -A, so anything untracked would be swept into the autofix commit.
    [ -z "$(git status --porcelain)" ] \
        || die "working tree is dirty (untracked files included); commit, stash or clean first"
    detect_host
    log "make fix-repo"
    PATH="$(dirname "$PY"):$PATH" make fix-repo
    if [ -z "$(git status --porcelain)" ]; then
        ok "nothing to regenerate"
    else
        # -A, not -u: fix-repo can create files (modular to modeling generation) and
        # `git add -u` would leave them untracked for build's clean check to miss.
        git add -A
        git commit -q -m "chore: repo-consistency autofix"
        ok "committed regeneration: $(git rev-parse --short HEAD)"
    fi
}

# ------------------------------------------------------------------------------ build

remove_worktree() {
    git worktree remove --force --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
    git worktree prune
}

cmd_strip_to_pr() {
    [ $# -eq 0 ] || die "strip-to-pr takes no flags"
    load_manifest
    assert_branch "$DEV_BRANCH"; assert_clean
    local dev; dev=$(git rev-parse HEAD)

    remove_worktree
    log "materializing the PR head in $WORKTREE"
    git worktree add -q --detach "$WORKTREE" "$dev"
    # A half-stripped leftover would otherwise be validated as if it were a real PR head.
    trap 'remove_worktree' ERR

    # git rm -r on a directory shared with upstream removes only tracked children, so
    # directory and file entries need no special casing.
    git -C "$WORKTREE" rm -r -q -- "${SPECS[@]}"
    git -C "$WORKTREE" commit -q -m "chore: drop development harnesses for upstream submission"
    trap - ERR

    git branch -f "$PR_BRANCH" "$(git -C "$WORKTREE" rev-parse HEAD)"
    ok "PR head rebuilt: $(git rev-parse --short "$PR_BRANCH")"
    cmd_verify_strip
}

# ----------------------------------------------------------------------------- verify

cmd_verify_strip() {
    [ $# -eq 0 ] || die "verify-strip takes no flags"
    load_manifest
    local dev pr
    dev=$(rev "$DEV_BRANCH"); pr=$(rev "$PR_BRANCH")
    [ -n "$dev" ] || die "no such branch: $DEV_BRANCH"
    [ -n "$pr" ] || die "no such branch: $PR_BRANCH (run 'pr-sync strip-to-pr')"

    log "invariants"

    local parent; parent=$(git rev-parse "$pr^")
    [ "$parent" = "$dev" ] || die "invariant 1: $PR_BRANCH^ is ${parent:0:10}, expected dev ${dev:0:10}"
    ok "1  pr head is dev plus one strip commit"

    local st p nonadditive=0
    while read -r st p; do
        [ -n "$st" ] || continue
        [ "$st" = A ] || { bad "invariant 2: non-additive change $st $p"; nonadditive=1; }
    done < <(git diff --name-status "$pr" "$dev")
    [ "$nonadditive" -eq 0 ] || die "invariant 2: the branches differ by more than added overlay files"
    ok "2  all differences are additions"

    local expected actual
    expected=$(expand_manifest "$dev" | LC_ALL=C sort -u)
    actual=$(git diff --name-only "$pr" "$dev" | LC_ALL=C sort -u)
    if [ "$expected" != "$actual" ]; then
        bad "invariant 3: overlay set mismatch"
        diff -u <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") >&2 || true
        note "< only in manifest = failed to strip;  > only in diff = LEAKED or DIVERGED"
        die "invariant 3 failed"
    fi
    ok "3  overlay set matches the manifest exactly"

    local base; base=$(git merge-base "$UPSTREAM/main" "$pr" 2>/dev/null || true)
    if [ -z "$base" ]; then
        warn "4  skipped: no merge-base with $UPSTREAM/main (run 'pr-sync fetch-remotes')"
        return 0
    fi
    local shipped re rc spec pats=()
    mapfile -t shipped < <(git diff --name-only "$base" "$pr")
    # An empty list would make git grep scan the whole tree, silently changing the check.
    [ ${#shipped[@]} -gt 0 ] || die "invariant 4: could not enumerate shipped files"
    # Patterns come from the manifest entries plus file stems. Deriving them from dirnames
    # would yield bare 'scripts' and 'tests/models/apertus1p5', which exist upstream and
    # match legitimate prose.
    for spec in "${SPECS[@]}"; do
        pats+=("$spec")
        case "$spec" in *.py) spec=${spec##*/}; pats+=("${spec%.py}") ;; esac
    done
    re=$(printf '%s\n' "${pats[@]}" | LC_ALL=C sort -u | sed 's/[].[^$\\*+?(){}|/]/\\&/g' | paste -sd'|' -)
    rc=0; git grep -I -n -E -- "$re" "$pr" -- "${shipped[@]}" >&2 || rc=$?
    [ "$rc" -eq 0 ] && die "invariant 4: a shipped file references a stripped path"
    # rc 1 is "no match"; anything else is a git error that must not read as success.
    [ "$rc" -eq 1 ] || die "invariant 4: git grep failed with rc=$rc"
    ok "4  no shipped file references a stripped path"
    note "shipped ${#shipped[@]} files; stripped $(printf '%s\n' "$expected" | wc -l) files"
}

# --------------------------------------------------------------------------- validate

LOG_DIR=""

# Results go to a file, not an array: run_tier is called from a background subshell for the
# concurrent lane and array writes there would be discarded, silently dropping a failure.
record() { printf '%s|%s|%s\n' "$1" "$2" "$3" >> "$LOG_DIR/results"; }
skip()   { note "skip $1: $2"; record "$1" SKIP "$2"; }

run_tier() {
    local name=$1 dir=$2; shift 2
    local logf="$LOG_DIR/$name.log" start rc=0
    log "$name"
    start=$(date +%s)
    ( cd "$dir" && "$@" ) >"$logf" 2>&1 || rc=$?
    local dur=$(( $(date +%s) - start ))
    if [ "$rc" -eq 0 ]; then
        ok "$name passed (${dur}s)"; record "$name" PASS "${dur}s"
    else
        bad "$name FAILED rc=$rc (${dur}s) -> $logf"
        tail -20 "$logf" >&2 || true
        record "$name" FAIL "rc=$rc, ${dur}s"
    fi
    return 0
}

# ap_testcase exits 0 when every case merely SKIPs, so the exit code alone cannot tell
# "9 passed" from "9 skipped". Parse the summary and treat an all-skip run as inconclusive.
run_ap() {
    local script=$1
    local name logf rc=0 line p f
    name=ap:$(basename "$script" .py)
    logf="$LOG_DIR/$name.log"
    ( cd "$ROOT" && "$PY" "$script" ) >"$logf" 2>&1 || rc=$?
    line=$(grep -oE '[0-9]+ passed, [0-9]+ failed, [0-9]+ skipped' "$logf" | tail -1 || true)
    if [ -z "$line" ]; then
        bad "$name: no summary line, crashed before reporting -> $logf"; record "$name" FAIL "no summary"; return 0
    fi
    read -r p _ f _ _ _ <<<"$line"
    if [ "$rc" -ne 0 ] || [ "$f" -gt 0 ]; then
        bad "$name: $line -> $logf"; record "$name" FAIL "$line"
    elif [ "$p" -eq 0 ]; then
        warn "$name: $line (nothing actually ran)"; record "$name" SKIP "$line"
    else
        ok "$name: $line"; record "$name" PASS "$line"
    fi
    return 0
}

wanted() { [ -z "$ONLY_TIER" ] || [ "$ONLY_TIER" = "$1" ]; }

# Records the skip and returns 1 when the tier is filtered out or cannot run here.
tier_ok() {
    wanted "$1" || return 1
    local why; why=$(tier_blocker "$1")
    [ -z "$why" ] || { skip "$1" "$why"; return 1; }
}

# The worktree must hold the PR head built from the current dev, or the checks tier would
# validate a leftover from a previous cycle and report it green.
worktree_current() {
    [ -d "$WORKTREE" ] || return 1
    [ -n "$(rev "$PR_BRANCH")" ] || return 1
    [ "$(git -C "$WORKTREE" rev-parse HEAD 2>/dev/null || true)" = "$(rev "$PR_BRANCH")" ] || return 1
    [ "$(rev "$PR_BRANCH^")" = "$(rev "$DEV_BRANCH")" ]
}

cmd_run_tests() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --only=*) ONLY_TIER=${1#*=} ;;
            *) die "run-tests: unknown flag $1" ;;
        esac
        shift
    done
    # An unrecognised tier would silently skip everything and exit 0, reading as success.
    if [ -n "$ONLY_TIER" ]; then
        printf '%s\n' "${TIERS[@]}" | grep -qxF "$ONLY_TIER" \
            || die "unknown tier '$ONLY_TIER' (one of: ${TIERS[*]})"
    fi
    detect_host
    LOG_DIR=$STATE_DIR/logs/$(date +%Y%m%d-%H%M%S)
    mkdir -p "$LOG_DIR"; : > "$LOG_DIR/results"
    log "logs -> $LOG_DIR"

    if wanted checks || wanted fast-pr; then
        if worktree_current; then
            if wanted checks; then
                run_tier checks "$WORKTREE" env PATH="$(dirname "$PY"):$PATH" make check-repo &
                local checks_pid=$!
            fi
            wanted fast-pr && run_tier fast-pr "$WORKTREE" "$PY" -m pytest "${TEST_PATHS[@]}" -q -n auto
            [ -n "${checks_pid:-}" ] && wait "$checks_pid"
        else
            skip checks "worktree missing or stale, run 'pr-sync strip-to-pr'"
        fi
    fi

    if [ "$(current_branch)" = "$DEV_BRANCH" ]; then
        tier_ok fast && run_tier fast "$ROOT" "$PY" -m pytest "${TEST_PATHS[@]}" -q -n auto
        tier_ok slow && run_tier slow "$ROOT" env RUN_SLOW=1 "$PY" -m pytest "${TEST_PATHS[@]}" -q -k Integration
        if tier_ok ap; then
            log "ap_testcase"
            local s
            # 01 and 02 build only the processor, so a processor break surfaces before any
            # weights load; 03 onward each load the full 8B model.
            for s in ap_testcase/0[1-7]_*.py; do run_ap "$s"; done
            if [ "$GPUS" -ge 1 ]; then for s in ap_testcase/09_*.py; do run_ap "$s"; done; fi
            if [ "$GPUS" -ge 2 ]; then
                for s in ap_testcase/08_*.py; do run_ap "$s"; done
            else
                skip ap:08 "needs 2 GPUs, have $GPUS"
            fi
        fi
    fi

    log "summary"
    local np=0 ns=0 nf=0 n st d c
    while IFS='|' read -r n st d; do
        case "$st" in
            PASS) c=$C_G; np=$((np+1)) ;;
            SKIP) c=$C_Y; ns=$((ns+1)) ;;
            FAIL) c=$C_R; nf=$((nf+1)) ;;
            *) continue ;;
        esac
        printf '  %s%s%s  %-20s %s\n' "$c" "$st" "$C_RST" "$n" "$d"
    done < "$LOG_DIR/results"
    printf '  %d passed, %d skipped, %d failed\n' "$np" "$ns" "$nf"
    # Zero tiers run is a false green, not a pass.
    [ $((np+ns+nf)) -gt 0 ] || die "run-tests ran no tiers"
    [ "$nf" -eq 0 ] || die "validation failed"
}

# ------------------------------------------------------------------------------- push

push_one() {
    local branch=$1 remote=$2 slot=$3
    local lease local_sha remote_sha force=()
    lease=$(cat "$STATE_DIR/lease_$slot" 2>/dev/null || true)
    local_sha=$(rev "$branch"); [ -n "$local_sha" ] || die "no such branch: $branch"
    git fetch -q "$remote" "$branch" || true
    remote_sha=$(rev "$remote/$branch")

    if [ -z "$remote_sha" ]; then
        log "creating $remote/$branch"
    elif [ "$remote_sha" = "$local_sha" ]; then
        ok "$remote/$branch already up to date"; return 0
    elif git merge-base --is-ancestor "$remote_sha" "$local_sha"; then
        log "fast-forwarding $remote/$branch"
    else
        [ -n "$lease" ] || die "$remote/$branch needs a force but no lease was recorded; run 'pr-sync fetch-remotes' first"
        [ "$remote_sha" = "$lease" ] \
            || die "$remote/$branch moved since cycle start (${lease:0:10} -> ${remote_sha:0:10}); refusing to force"
        # --cherry-pick --right-only, not a plain count: after a rebase every remote commit
        # looks unreachable, and a naive count would demand an override every single cycle.
        local dropping
        dropping=$(git rev-list --count --cherry-pick --right-only "$local_sha...$remote_sha")
        [ "$dropping" -eq 0 ] || warn "force drops $dropping commit(s) present only on $remote/$branch"
        log "force-pushing $remote/$branch (${remote_sha:0:10} -> ${local_sha:0:10})"
        force=(--force-with-lease="refs/heads/$branch:$lease")
    fi
    git push "${force[@]}" "$remote" "refs/heads/$branch:refs/heads/$branch"
    printf '%s\n' "$local_sha" > "$STATE_DIR/lease_$slot"
}

cmd_push() {
    case "${1:-}" in
        --dev) push_one "$DEV_BRANCH" "$DEV_REMOTE" dev ;;
        --pr)  push_one "$PR_BRANCH" "$PR_REMOTE" pr ;;
        *) die "push needs --dev or --pr" ;;
    esac
}

# -------------------------------------------------------------------------------- all

cmd_full_cycle() {
    [ $# -eq 0 ] || die "full-cycle takes no flags"
    cmd_check_host
    cmd_fetch_remotes
    cmd_rebase_dev
    cmd_autofix_dev
    cmd_strip_to_pr
    # One validate, after build: it covers the PR lane (in the worktree) and the dev lane
    # (this tree) in a single pass, since build leaves us on dev.
    cmd_run_tests
    log "done. push is manual: pr-sync push --dev, pr-sync push --pr"
}

# ------------------------------------------------------------------------------- main

# Marker-based so editing the header cannot break --help.
usage() { sed -n '/^# USAGE$/,/^# END USAGE$/p' "$SELF" | sed -e '1d' -e '$d' -e 's/^#\{0,1\} \{0,1\}//'; }

SUB=${1:-}; shift || true
case "$SUB" in
    check-host)    cmd_check_host "$@" ;;
    status)        cmd_status "$@" ;;
    fetch-remotes) cmd_fetch_remotes "$@" ;;
    rebase-dev)    cmd_rebase_dev "$@" ;;
    autofix-dev)   cmd_autofix_dev "$@" ;;
    strip-to-pr)   cmd_strip_to_pr "$@" ;;
    verify-strip)  cmd_verify_strip "$@" ;;
    run-tests)     cmd_run_tests "$@" ;;
    push)          cmd_push "$@" ;;
    full-cycle)    cmd_full_cycle "$@" ;;
    ""|-h|--help)  usage ;;
    *) die "unknown subcommand: $SUB (try --help)" ;;
esac
