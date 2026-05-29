from __future__ import annotations

import argparse
from pathlib import Path

from app.pipeline.runner import ReceiptPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local receipt to Excel automation")
    parser.add_argument("input", type=Path, help="Image file or directory of receipt images")
    parser.add_argument("--output", type=Path, default=Path("outputs/receipts.xlsx"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = ReceiptPipeline()
    results = pipeline.process_path(args.input, args.output)
    accepted = sum(result.status.value == "accepted" for result in results)
    review = sum(result.status.value == "needs_review" for result in results)
    failed = sum(result.status.value == "failed" for result in results)
    print(f"Processed {len(results)} receipt(s): {accepted} accepted, {review} review, {failed} failed")
    print(f"Workbook: {args.output}")


if __name__ == "__main__":
    main()
