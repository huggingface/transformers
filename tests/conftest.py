# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import sys
import warnings
from os.path import abspath, dirname, join


# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)


# silence FutureWarning warnings in tests since often we can't act on them until
# they become normal warnings - i.e. the tests still need to test the current functionality
warnings.simplefilter(action="ignore", category=FutureWarning)

def pytest_configure(config):
    config.addinivalue_line("markers", "is_pipeline_test: mark test to run only when pipeline are tested")
    config.addinivalue_line(
        "markers", "is_pt_tf_cross_test: mark test to run only when PT and TF interactions are tested"
    )

# generate dedicated reports
import pytest  # noqa


report_files = dict(
    #    failures="tests_failures.txt",
    #    skipped="tests_skipped.txt",
    durations="tests_durations.txt",
    short_summary="tests_short_summary.txt",
    summary_errors="tests_errors.txt",
    summary_failures="tests_failures.txt",
    summary_warnings="tests_warnings.txt",
    summary_passes="tests_passes.txt",
    summary_stats="tests_stats.txt",
)


# XXX: appending to files, so need to reset them before each pytest run
# XXX: do we need to lock files to handle xdist?

# @pytest.hookimpl(tryfirst=True, hookwrapper=True)
# def pytest_runtest_makereport(item, call):
#     # execute all other hooks to obtain the report object
#     outcome = yield
#     report = outcome.get_result()

#     # failing tests
#     if report.when == "call" and report.failed:
#         with open(report_files["failures"], "a") as f:
#             # f.write(report.shortreprtext + "\n")
#             f.write(report.longreprtext + "\n")

#     # # skipped tests
#     # if report.skipped and report.outcome == "skipped":
#     #     with open(report_files["skipped"], "a") as f:
#     #         f.write(report.longreprtext + "\n")

from _pytest.config import create_terminal_writer  # noqa


def pytest_terminal_summary(terminalreporter):
    tr = terminalreporter
    config = tr.config
    orig_writer = config.get_terminal_writer()

    # custom durations report
    # note: no longer need to add --durations=XX to get it
    # adapted from https://github.com/pytest-dev/pytest/blob/master/src/_pytest/runner.py#L66
    dlist = []
    for replist in tr.stats.values():
        for rep in replist:
            if hasattr(rep, "duration"):
                dlist.append(rep)
    if dlist:
        dlist.sort(key=lambda x: x.duration, reverse=True)
        with open(report_files["durations"], "w") as f:
            # XXX: add a cutoff - having a million 0.000 entries
            f.write("slowest durations\n")
            for i, rep in enumerate(dlist):
                f.write(f"{rep.duration:02.2f}s {rep.when:<8} {rep.nodeid}\n")

    # use ready made report funcs, we are just hijacking the filehandle to log to a separate file each

    with open(report_files["summary_errors"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_errors()

    with open(report_files["summary_failures"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_failures()

    with open(report_files["summary_warnings"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_warnings()

    with open(report_files["summary_passes"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_passes()

    with open(report_files["short_summary"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.short_test_summary()

    with open(report_files["summary_stats"], "w") as f:
        tr._tw = create_terminal_writer(config, f)
        tr.summary_stats()

    # restore the writer
    tr._tw = orig_writer


