"""Command-line runner for checking dataset preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module1.preprocessing.preprocessing import (
    build_preprocessing_summary,
    load_dataset,
    preprocess_dataframe,
    save_preprocessed_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the synthetic answer dataset.")
    parser.add_argument("input", type=Path, help="Path to the input .xlsx, .xls, or .csv dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("module1/outputs/preprocessed_dataset.csv"),
        help="Path for the cleaned CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dataframe = load_dataset(args.input)
    processed_dataframe = preprocess_dataframe(raw_dataframe)
    save_preprocessed_dataset(processed_dataframe, args.output)

    summary = build_preprocessing_summary(processed_dataframe)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"\nSaved cleaned dataset to: {args.output}")


if __name__ == "__main__":
    main()
