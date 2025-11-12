#!/usr/bin/env python3
"""
Simple interactive CLI to probe IdeaPipeline predictions.

Usage:
    python idea/examples/interactive_cli.py
"""

import sys
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from idea import IdeaPipeline

def main():
    pipe = IdeaPipeline()
    pipe.initialize()
    print("Interactive IdeaPipeline (type 'quit' to exit)")

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        result = pipe.run(question, fusion_top_k=3)
        tag = result.get("primary_tag")
        print(f"Top tag: {tag}")
        print("Candidates:")
        for idx, cand in enumerate(result.get("candidates", []), 1):
            print(f"  {idx}. {cand['tag']} (score={cand['final_score']:.3f})")
        print()

if __name__ == "__main__":
    sys.exit(main())
