"""Excel helper utilities for the local assistant.

Provides simple, safe helpers to load CSVs into Excel workbooks and apply
simple step-by-step calculation instructions of the form:

    NewCol = ColA * ColB + 10

The apply_steps function accepts a list of such expressions and evaluates them
with pandas.eval in the DataFrame context.
"""
from typing import List
import os
import pandas as pd


def load_csv_to_dataframe(csv_path: str) -> pd.DataFrame:
    """Load a CSV into a pandas DataFrame."""
    return pd.read_csv(csv_path)


def write_dataframe_to_excel(df: pd.DataFrame, excel_path: str, sheet_name: str = "Sheet1", mode: str = "w") -> None:
    """Write DataFrame to an Excel file. mode 'w' creates/overwrites file, 'a' appends sheet when possible."""
    # Use openpyxl engine for xlsx
    if mode == "w" or not os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        # append a new sheet to existing workbook
        from openpyxl import load_workbook

        book = load_workbook(excel_path)
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a") as writer:
            writer.book = book
            df.to_excel(writer, index=False, sheet_name=sheet_name)


def apply_steps(df: pd.DataFrame, steps: List[str]) -> pd.DataFrame:
    """Apply a list of calculation steps to the DataFrame.

    Each step should be a string representing an assignment using column
    names, e.g. 'Total = Price * Quantity'. The expression on the right-hand
    side is evaluated with pandas.eval.
    """
    for step in steps:
        step = step.strip()
        if not step:
            continue
        if "=" not in step:
            raise ValueError(f"Unsupported step format (expected 'NewCol = expr'): {step}")
        left, right = step.split("=", 1)
        col = left.strip()
        expr = right.strip()
        # Evaluate expression with pandas.eval in the context of the DataFrame
        # To allow column names with spaces, users should reference them like `df['Col Name']`.
        try:
            df[col] = pd.eval(expr, engine="python", local_dict={**{c: df[c] for c in df.columns}})
        except Exception as e:
            # Re-raise with context
            raise RuntimeError(f"Failed to apply step '{step}': {e}")
    return df


def load_sheet_from_excel(excel_path: str, sheet_name: str = "Sheet1"):
    """Load a specific sheet from an Excel workbook into a DataFrame."""
    return pd.read_excel(excel_path, sheet_name=sheet_name)


def filter_df(df: pd.DataFrame, expression: str) -> pd.DataFrame:
    """Filter DataFrame rows using a pandas.eval-compatible boolean expression.

    Example: "Quantity > 10 and Country == 'US'"
    """
    try:
        mask = pd.eval(expression, engine="python", local_dict={**{c: df[c] for c in df.columns}})
        return df[mask]
    except Exception as e:
        raise RuntimeError(f"Failed to filter DataFrame with '{expression}': {e}")


def groupby_aggregate(df: pd.DataFrame, by: List[str], aggs: dict) -> pd.DataFrame:
    """Group by columns and aggregate.

    `aggs` is a dict mapping column -> aggregation function or list of functions.
    Example: groupby_aggregate(df, ['Country'], {'Price': 'sum', 'Quantity': 'mean'})
    """
    try:
        return df.groupby(by).agg(aggs).reset_index()
    except Exception as e:
        raise RuntimeError(f"Failed to groupby/aggregate: {e}")


def rename_columns(df: pd.DataFrame, renames: dict) -> pd.DataFrame:
    """Rename columns. `renames` is a dict old_name -> new_name."""
    return df.rename(columns=renames)


def drop_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Drop columns from DataFrame."""
    return df.drop(columns=cols)


def apply_batch_steps(df: pd.DataFrame, steps: List[str]) -> pd.DataFrame:
    """Apply multiple semicolon-separated steps sequentially.

    Steps may be passed as a list or as a single string containing semicolon-separated expressions.
    Each step uses the same assignment format as `apply_steps`.
    """
    expanded = []
    for s in steps:
        if isinstance(s, str) and ";" in s:
            expanded.extend([x.strip() for x in s.split(";") if x.strip()])
        else:
            expanded.append(s)
    return apply_steps(df, expanded)


def preview_apply_steps(df: pd.DataFrame, steps: List[str], n: int = 5) -> pd.DataFrame:
    """Return the top `n` rows of the DataFrame after applying steps without modifying the input DataFrame.

    Useful for dry-run / preview before writing changes to disk.
    """
    df_copy = df.copy()
    df_copy = apply_steps(df_copy, steps)
    return df_copy.head(n)
