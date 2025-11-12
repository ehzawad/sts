#!/usr/bin/env python
"""
Evaluate IdeaPipeline on a labeled CSV (question, tag).

Usage:
    python idea/evals/eval_pipeline.py --data idea/datasets/sts_eval.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from idea import IdeaPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IdeaPipeline accuracy on a CSV file.")
    parser.add_argument(
        "--data",
        default="idea/datasets/sts_eval.csv",
        help="CSV file with columns: question, tag (default: idea/datasets/sts_eval.csv)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="How many fused candidates to consider for top-k accuracy (default: 1)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Eval CSV not found: {data_path}")

    # Load eval rows
    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "question" not in reader.fieldnames or "tag" not in reader.fieldnames:
            raise ValueError("CSV must contain 'question' and 'tag' columns")
        for line in reader:
            rows.append({"question": line["question"], "tag": line["tag"]})

    if not rows:
        print("No rows found in eval CSV.")
        return

    # Initialize pipeline
    pipeline = IdeaPipeline()
    pipeline.initialize()

    total = len(rows)
    top1_correct = 0
    topk_correct = 0
    misses_top1 = []
    misses_topk = []

    for idx, row in enumerate(rows, 1):
        question = row["question"]
        expected_tag = row["tag"]
        result = pipeline.run(question, fusion_top_k=args.top_k)
        candidates = result.get("candidates", [])
        predicted_tag = result.get("primary_tag")

        if predicted_tag == expected_tag:
            top1_correct += 1
        else:
            misses_top1.append((question, expected_tag, predicted_tag))

        candidate_tags = [cand["tag"] for cand in candidates[:args.top_k]]
        if expected_tag in candidate_tags:
            topk_correct += 1
        else:
            misses_topk.append((question, expected_tag, candidate_tags))

        if idx % 100 == 0 or idx == total:
            print(f"Processed {idx}/{total}")

    top1_acc = 100.0 * top1_correct / total
    topk_acc = 100.0 * topk_correct / total

    print("\nEvaluation complete")
    print(f"Total rows: {total}")
    print(f"Top-1 accuracy: {top1_acc:.2f}%")
    print(f"Top-{args.top_k} accuracy: {topk_acc:.2f}%")
    print(f"Top-1 misses: {len(misses_top1)} (showing up to 5)")
    for q, exp, pred in misses_top1[:5]:
        print(f"- Expected {exp}, got {pred}; question snippet: {q[:60]}...")
    print(f"\nTop-{args.top_k} misses: {len(misses_topk)} (showing up to 5)")
    for q, exp, preds in misses_topk[:5]:
        print(f"- Expected {exp}, top-{args.top_k} candidates {preds}; snippet: {q[:60]}...")


if __name__ == "__main__":
    main()
