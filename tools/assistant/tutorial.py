"""Tutorial script demonstrating assistant Excel workflow.

This is a non-interactive script that shows how to use the helpers:
- load CSV
- apply steps
- save to Excel
"""
from excel_helper import load_csv_to_dataframe, apply_steps, write_dataframe_to_excel


def demo():
    csv_path = "example.csv"
    excel_path = "example_output.xlsx"
    sheet = "Sheet1"

    print(f"Loading {csv_path}")
    df = load_csv_to_dataframe(csv_path)
    print(df.head())

    steps = [
        "Total = Price * Quantity",
        "VAT = Total * 0.2",
    ]
    print("Applying steps:", steps)
    df = apply_steps(df, steps)
    print(df.head())

    print(f"Writing to {excel_path} {sheet}")
    write_dataframe_to_excel(df, excel_path, sheet)


if __name__ == '__main__':
    demo()
