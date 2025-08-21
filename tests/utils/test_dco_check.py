from __future__ import annotations

from transformers.utils.dco_check import has_dco_signoff


def test_has_dco_positive():
    msg = """
    Fix bug in foo

    Signed-off-by: Alice <alice@example.com>
    """
    assert has_dco_signoff(msg) is True


def test_has_dco_negative():
    msg = """
    Fix bug in foo

    Co-authored-by: Bob <bob@example.com>
    """
    assert has_dco_signoff(msg) is False


def test_has_dco_empty():
    assert has_dco_signoff("") is False
