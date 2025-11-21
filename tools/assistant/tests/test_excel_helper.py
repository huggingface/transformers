import pandas as pd
from excel_helper import apply_steps, preview_apply_steps


def test_apply_simple_step(tmp_path):
    df = pd.DataFrame({"Price": [10, 5], "Quantity": [2, 3]})
    res = apply_steps(df.copy(), ["Total = Price * Quantity"])
    assert "Total" in res.columns
    assert list(res["Total"]) == [20, 15]


def test_preview_apply_steps():
    df = pd.DataFrame({"Price": [10, 5], "Quantity": [2, 3]})
    preview = preview_apply_steps(df, ["Total = Price * Quantity"], n=2)
    assert preview.shape[0] == 2
    assert "Total" in preview.columns
