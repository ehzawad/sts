"""
CLI shim for training / fine-tuning the 195-tag classifier used by idea.pipeline.

The heavy lifting is delegated to the UnifiedTagClassifierTrainer defined inside
idea/training/unified_tag_classifier_trainer.py, keeping all plumbing local to
the idea package.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
import types

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from idea.training.unified_tag_classifier_trainer import UnifiedTagClassifierTrainer

LOG = logging.getLogger("idea.training.tag_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 195-tag classifier used by idea.pipeline"
    )
    parser.add_argument("--models", default="idea/models/tag_classifier", help="Output directory for checkpoints")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-large-instruct", help="SentenceTransformer backbone")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs (trainer uses early stopping)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=12, help="Early-stopping patience")
    parser.add_argument("--ngram-mode", choices=("manual", "auto"), default="manual", help="Which featurizer output to consume")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    models_dir = Path(args.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Output directory: %s", models_dir.resolve())
    LOG.info("Embedding model: %s", args.embedding_model)
    LOG.info("Using %s n-gram features", args.ngram_mode)

    trainer = UnifiedTagClassifierTrainer(embedding_model=args.embedding_model)

    # Override the featurizer selection without editing the upstream trainer.
    original_loader = trainer.load_ngram_features

    def load_with_custom_mode(self, active_ngram_file="manual"):
        return original_loader(args.ngram_mode)

    trainer.load_ngram_features = types.MethodType(load_with_custom_mode, trainer)

    df_train, df_eval = trainer.load_data()
    LOG.info("Training rows: %d | Eval rows: %d | Tags: %d", len(df_train), len(df_eval), df_train['tag'].nunique())

    start_time = time.time()
    model = trainer.train(
        df_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience
    )
    training_minutes = (time.time() - start_time) / 60

    trainer.save_model(model, models_dir)
    LOG.info("Training finished in %.2f minutes", training_minutes)


if __name__ == "__main__":
    main()
