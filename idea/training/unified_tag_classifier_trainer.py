# -*- coding: utf-8 -*-
"""
Unified Tag Classifier Training (tag-only)

Trains one model that takes question embeddings plus per-tag n-gram features
to predict across all 195 tags. Cluster information has been removed.
"""

import json
import os
import logging
import time
from pathlib import Path
from datetime import datetime
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from idea.utils.model_cache import get_shared_embedding_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedTagClassifier(nn.Module):
    """Two-branch (embedding + pattern) tag classifier."""

    def __init__(self, embedding_dim=1024, pattern_dim=390, num_tags=195, dropout=0.5):
        super().__init__()

        self.emb_fc1 = nn.Linear(embedding_dim, 512)
        self.emb_bn1 = nn.BatchNorm1d(512)
        self.emb_fc2 = nn.Linear(512, 256)
        self.emb_bn2 = nn.BatchNorm1d(256)

        self.pattern_fc1 = nn.Linear(pattern_dim, 128)
        self.pattern_bn1 = nn.BatchNorm1d(128)
        self.pattern_fc2 = nn.Linear(128, 64)
        self.pattern_bn2 = nn.BatchNorm1d(64)

        fusion_input_dim = 256 + 64
        self.fusion_fc1 = nn.Linear(fusion_input_dim, 128)
        self.fusion_bn1 = nn.BatchNorm1d(128)
        self.output = nn.Linear(128, num_tags)

        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, patterns):
        emb = F.relu(self.emb_bn1(self.emb_fc1(embeddings)))
        emb = self.dropout(emb)
        emb = F.relu(self.emb_bn2(self.emb_fc2(emb)))
        emb = self.dropout(emb)

        pat = F.relu(self.pattern_bn1(self.pattern_fc1(patterns)))
        pat = self.dropout(pat)
        pat = F.relu(self.pattern_bn2(self.pattern_fc2(pat)))
        pat = self.dropout(pat)

        fused = torch.cat([emb, pat], dim=1)
        fused = F.relu(self.fusion_bn1(self.fusion_fc1(fused)))
        fused = self.dropout(fused)
        return self.output(fused)


class TagDataset(Dataset):
    """Dataset for unified tag classification"""

    def __init__(self, embeddings, patterns, labels):
        self.embeddings = embeddings
        self.patterns = patterns
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.embeddings[idx]),
            torch.FloatTensor(self.patterns[idx]),
            torch.LongTensor([self.labels[idx]])[0]
        )


class UnifiedTagClassifierTrainer:
    """Trainer for the tag-only unified classifier"""

    def __init__(self, embedding_model='intfloat/multilingual-e5-large-instruct'):
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.device = None
        self.tag_encoder = LabelEncoder()  # For tags (all ~195 tags)

    @staticmethod
    def _preview_ngrams(ngrams, limit=5):
        """Return a compact, human-readable snapshot of a set of n-grams"""
        if not ngrams:
            return "none"

        sorted_items = sorted(ngrams)
        if len(sorted_items) <= limit:
            return '; '.join(sorted_items)

        remaining = len(sorted_items) - limit
        return '; '.join(sorted_items[:limit]) + f"; ... (+{remaining} more)"

    def load_data(self):
        """Load training/eval splits (question + tag only)"""
        logger.info("Loading training and eval data...")

        # Paths
        component_dir = Path(__file__).parent.parent
        datasets_dir = component_dir / "datasets"

        # Load training data
        train_file = datasets_dir / "sts_train.csv"
        df_train = pd.read_csv(train_file)
        logger.info(f"  Train: {len(df_train)} questions (tags: {df_train['tag'].nunique()})")

        # Load eval data
        eval_file = datasets_dir / "sts_eval.csv"
        df_eval = pd.read_csv(eval_file)
        logger.info(f"  Eval: {len(df_eval)} questions (tags: {df_eval['tag'].nunique()})")

        # Show tag distribution (top tags)
        logger.info(f"\n  Top 10 tags (train):")
        tag_counts = df_train['tag'].value_counts()
        for tag, count in tag_counts.head(10).items():
            logger.info(f"    {tag:50s}: {count:4d}")

        return df_train, df_eval

    def load_ngram_features(self, active_ngram_file='manual'):
        """
        Load n-gram features from JSON file

        Args:
            active_ngram_file: 'manual' or 'auto' (default: 'manual')

        Returns:
            Tuple of (tag_patterns, tags_sorted)
        """
        features_dir = Path(__file__).parent.parent / "datasets" / "features"
        if not features_dir.exists():
            raise FileNotFoundError(f"Features directory missing: {features_dir}")

        ngram_file = f"{active_ngram_file}_ngrams.json"
        features_path = features_dir / ngram_file

        logger.info(f"Loading n-gram features from: {features_path}")

        if not features_path.exists():
            raise FileNotFoundError(
                f"\nN-gram features file not found: {features_path}\n"
                f"Please run: python idea/featurizer/generate_features.py --top-k 20 [--auto-clean]\n"
                f"and ensure the JSON files exist under idea/datasets/features/."
            )

        with open(features_path, 'r', encoding='utf-8') as f:
            features = json.load(f)

        # Convert to tag_patterns dict for fast lookup
        tag_patterns = {}
        for tag_name, tag_data in features["tags"].items():
            tag_patterns[tag_name] = {
                "trigrams": set(item["ngram"] for item in tag_data["trigrams"]),
                "fourgrams": set(item["ngram"] for item in tag_data["fourgrams"])
            }

        tags_sorted = sorted(tag_patterns.keys())

        logger.info(f"  Loaded features for {len(tags_sorted)} tags")
        logger.info(f"  Pattern dimension: {len(tags_sorted) * 2}")

        # Show sample unique n-grams per tag
        for tag in tags_sorted[:3]:
            trigram_set = tag_patterns[tag]['trigrams']
            fourgram_set = tag_patterns[tag]['fourgrams']
            logger.info(
                f"    {tag}: {len(trigram_set)} unique trigrams "
                f"| sample: {self._preview_ngrams(trigram_set)}"
            )
            logger.info(
                f"           {len(fourgram_set)} unique 4-grams "
                f"| sample: {self._preview_ngrams(fourgram_set)}"
            )

        return tag_patterns, tags_sorted

    def extract_ngram_words(self, text, n):
        """
        Extract word-based n-grams (not character n-grams)

        Args:
            text: Input text
            n: N-gram size

        Returns:
            Set of n-grams
        """
        words = text.split()
        if len(words) < n:
            return set()
        return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}

    def compute_pattern_features(self, questions, tag_patterns, tags_sorted):
        """
        Compute per-tag n-gram matching features

        Args:
            questions: List of question strings
            tag_patterns: Dict mapping tag → {trigrams: set, fourgrams: set}
            tags_sorted: Sorted list of tag names

        Returns:
            features_array of shape (n_questions, 390) = (n_questions, 2 * 195)
        """
        logger.info("Computing per-tag n-gram matching features (390-dim)...")

        features = []
        for idx, question in enumerate(tqdm(questions, desc="Extracting patterns", disable=False)):
            # Extract n-grams from question
            q_trigrams = self.extract_ngram_words(question, 3)
            q_fourgrams = self.extract_ngram_words(question, 4)

            # Match against each tag's n-grams
            feature_vec = []
            sample_trigram_matches = set()
            sample_fourgram_matches = set()

            for tag in tags_sorted:
                # Count matches
                trigram_overlap = q_trigrams & tag_patterns[tag]["trigrams"]
                fourgram_overlap = q_fourgrams & tag_patterns[tag]["fourgrams"]

                trigram_matches = len(trigram_overlap)
                fourgram_matches = len(fourgram_overlap)
                feature_vec.extend([trigram_matches, fourgram_matches])

                if idx < 3:
                    sample_trigram_matches |= trigram_overlap
                    sample_fourgram_matches |= fourgram_overlap

            features.append(feature_vec)

            if idx < 3:
                logger.info(f"\n  Sample [{idx}] Question: {question[:70]}...")
                logger.info(
                    f"           Unique question trigrams ({len(q_trigrams)}): "
                    f"{self._preview_ngrams(q_trigrams)}"
                )
                logger.info(
                    f"           Unique question 4-grams ({len(q_fourgrams)}): "
                    f"{self._preview_ngrams(q_fourgrams)}"
                )
                logger.info(
                    f"           Matched feature trigrams ({len(sample_trigram_matches)}): "
                    f"{self._preview_ngrams(sample_trigram_matches)}"
                )
                logger.info(
                    f"           Matched feature 4-grams ({len(sample_fourgram_matches)}): "
                    f"{self._preview_ngrams(sample_fourgram_matches)}"
                )

        features_array = np.array(features, dtype=np.float32)

        # Normalize
        mean = features_array.mean(axis=0, keepdims=True)
        std = features_array.std(axis=0, keepdims=True) + 1e-7
        features_array = (features_array - mean) / std

        logger.info(f"  Pattern features shape: {features_array.shape}")
        logger.info(f"  Expected shape: (n_questions, {len(tags_sorted) * 2})")

        return features_array, mean, std

    def generate_embeddings(self, questions):
        """Generate sentence embeddings using shared model cache"""
        logger.info("Generating embeddings using shared model cache...")

        # Device selection: CUDA → MPS → CPU
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        logger.info(f"  Device: {self.device}")

        # Load model from shared cache (avoids duplicate 1.2GB model)
        logger.info(f"  Loading shared embedding model: {self.embedding_model_name}")
        self.embedding_model = get_shared_embedding_model(self.embedding_model_name)

        # Generate embeddings
        batch_size = 128 if self.device == 'cuda' else 32
        embeddings = self.embedding_model.encode(
            questions,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=batch_size
        )

        logger.info(f"  Embeddings shape: {embeddings.shape}")
        return embeddings

    def train(self, df_train, epochs=200, batch_size=64, lr=0.001, patience=15):
        """
        Train unified tag classifier with early stopping

        Args:
            df_train: Training DataFrame
            epochs: Maximum number of epochs (default: 200)
            batch_size: Batch size (default: 64)
            lr: Learning rate (default: 0.001)
            patience: Early stopping patience - stop if no improvement for this many epochs (default: 15)
        """
        logger.info("\nTraining unified tag classifier...")
        logger.info(f"  Max epochs: {epochs}, Patience: {patience}")

        # Encode tag labels (all ~180 tags)
        tag_labels = self.tag_encoder.fit_transform(df_train['tag'].values)
        num_tags = len(self.tag_encoder.classes_)
        logger.info(f"  Number of tag classes: {num_tags}")

        # Load n-gram features
        tag_patterns, tags_sorted = self.load_ngram_features(active_ngram_file='manual')

        # Store as instance variables for reuse in evaluate()
        self.tag_patterns = tag_patterns
        self.tags_sorted = tags_sorted

        # Validate that loaded tags match training data
        training_tags = set(df_train['tag'].unique())
        loaded_tags = set(tags_sorted)
        missing_in_json = training_tags - loaded_tags
        if missing_in_json:
            raise ValueError(
                f"\n{len(missing_in_json)} tags in training data not found in n-gram features JSON:\n"
                f"{sorted(missing_in_json)}\n"
                f"Please regenerate features (e.g., python idea/featurizer/generate_features.py) and ensure they're stored in idea/datasets/features/."
            )

        # Generate features
        embeddings = self.generate_embeddings(df_train['question'].tolist())
        patterns, self.pattern_mean, self.pattern_std = self.compute_pattern_features(
            df_train['question'].tolist(), tag_patterns, tags_sorted
        )

        # Train/val split (stratified by tag to ensure balanced representation)
        X_emb_train, X_emb_val, X_pat_train, X_pat_val, y_train, y_val = train_test_split(
            embeddings, patterns, tag_labels,
            test_size=0.2,
            random_state=42,
            stratify=tag_labels
        )

        logger.info(f"  Train: {len(y_train)}, Val: {len(y_val)}")

        # Create datasets
        train_dataset = TagDataset(X_emb_train, X_pat_train, y_train)
        val_dataset = TagDataset(X_emb_val, X_pat_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Model
        pattern_dim = len(tags_sorted) * 2  # 2 features per tag
        logger.info(f"  Pattern dimension: {pattern_dim}")

        model = UnifiedTagClassifier(
            embedding_dim=1024,
            pattern_dim=pattern_dim,
            num_tags=num_tags,
            dropout=0.5
        )
        model = model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Model parameters: {total_params:,} (trainable: {trainable_params:,})")

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        logger.info("\nTraining...")
        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for emb, pat, labels_batch in train_loader:
                emb = emb.to(self.device)
                pat = pat.to(self.device)
                labels_batch = labels_batch.to(self.device)

                optimizer.zero_grad()
                outputs = model(emb, pat)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels_batch.size(0)
                train_correct += predicted.eq(labels_batch).sum().item()

            train_acc = 100.0 * train_correct / train_total

            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for emb, pat, labels_batch in val_loader:
                    emb = emb.to(self.device)
                    pat = pat.to(self.device)
                    labels_batch = labels_batch.to(self.device)

                    outputs = model(emb, pat)
                    loss = criterion(outputs, labels_batch)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels_batch.size(0)
                    val_correct += predicted.eq(labels_batch).sum().item()

            val_acc = 100.0 * val_correct / val_total

            # Scheduler step
            scheduler.step(val_loss)

            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            # Early stopping check (based on validation loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                self.best_model_state = model.state_dict()
                patience_counter = 0
                logger.info(f"  Epoch {epoch+1:3d}/{epochs}: "
                           f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                           f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% ← NEW BEST")
            else:
                patience_counter += 1
                # Log progress every 5 epochs or when patience increases
                if (epoch + 1) % 5 == 0:
                    logger.info(f"  Epoch {epoch+1:3d}/{epochs}: "
                               f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                               f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% (patience: {patience_counter}/{patience})")

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"\n  ⚠ Early stopping triggered at epoch {epoch + 1}")
                logger.info(f"  Best model was at epoch {best_epoch} with val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.2f}%")
                break

        # Store training metadata
        self.trained_epochs = epoch + 1  # Actual epochs trained
        self.best_epoch = best_epoch
        self.best_val_acc = best_val_acc
        self.early_stopped = (epoch + 1) < epochs

        logger.info(f"\n  Training Summary:")
        logger.info(f"    Epochs trained: {self.trained_epochs}/{epochs}")
        logger.info(f"    Best epoch: {best_epoch}")
        logger.info(f"    Best val loss: {best_val_loss:.4f}")
        logger.info(f"    Best val accuracy: {best_val_acc:.2f}%")
        if self.early_stopped:
            logger.info(f"    Early stopped: Yes (patience reached)")
        else:
            logger.info(f"    Early stopped: No (max epochs reached)")

        # Load best model
        model.load_state_dict(self.best_model_state)
        return model

    def evaluate(self, model, df_eval):
        """Evaluate on eval set"""
        logger.info("\nEvaluating on eval set...")

        # Encode labels
        tag_labels = self.tag_encoder.transform(df_eval['tag'].values)

        # Generate features (reuse existing embedding model to avoid OOM)
        logger.info("Generating embeddings (using cached model)...")
        batch_size = 128 if self.device == 'cuda' else 32
        embeddings = self.embedding_model.encode(
            df_eval['question'].tolist(),
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=batch_size
        )
        logger.info(f"  Embeddings shape: {embeddings.shape}")

        patterns = self.compute_pattern_features(
            df_eval['question'].tolist(),
            self.tag_patterns,
            self.tags_sorted
        )[0]

        # Normalize patterns using training stats
        patterns = (patterns - self.pattern_mean) / self.pattern_std

        # Create dataset
        eval_dataset = TagDataset(embeddings, patterns, tag_labels)
        eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for emb, pat, labels_batch in eval_loader:
                emb = emb.to(self.device)
                pat = pat.to(self.device)
                labels_batch = labels_batch.to(self.device)

                outputs = model(emb, pat)
                _, predicted = outputs.max(1)

                total += labels_batch.size(0)
                correct += predicted.eq(labels_batch).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        overall_accuracy = 100.0 * correct / total
        logger.info(f"\n  Overall eval accuracy: {overall_accuracy:.2f}% ({correct}/{total})")

        return overall_accuracy, all_predictions, all_labels, {}

    def save_model(self, model, output_dir):
        """Save unified model and metadata"""
        logger.info("\nSaving unified model...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = output_dir / "unified_tag_classifier.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'tag_encoder_classes': self.tag_encoder.classes_.tolist(),
            'pattern_mean': self.pattern_mean.tolist(),
            'pattern_std': self.pattern_std.tolist(),
            'embedding_model': self.embedding_model_name,
            'num_tags': len(self.tag_encoder.classes_),
            # Training info
            'trained_epochs': self.trained_epochs,
            'best_epoch': self.best_epoch,
            'best_val_acc': self.best_val_acc,
            'early_stopped': self.early_stopped
        }, model_file)

        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        logger.info(f"  Model saved: {model_file.name} ({file_size_mb:.2f} MB)")

        # Save metadata
        metadata = {
            'model_type': 'unified_tag_classifier',
            'embedding_model': self.embedding_model_name,
            'num_tags': len(self.tag_encoder.classes_),
            'tags': self.tag_encoder.classes_.tolist(),
            # Training info
            'trained_epochs': self.trained_epochs,
            'best_epoch': self.best_epoch,
            'best_val_acc': float(self.best_val_acc),
            'early_stopped': self.early_stopped,
            'created_at': datetime.now().isoformat()
        }

        metadata_file = output_dir / "unified_tag_classifier_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"  Metadata saved: {metadata_file.name}")
        logger.info(f"\n  All artifacts saved to: {output_dir}")


def main():
    """CLI entry point for tag-only unified classifier training"""
    print("=" * 80)
    print("UNIFIED TAG CLASSIFIER TRAINING (TAG-ONLY)")
    print("=" * 80)

    component_dir = Path(__file__).parent.parent
    models_dir = component_dir / "models" / "classifier"

    print(f"\nOutput directory: {models_dir}")
    print("\nThis trains a unified model for all 195 tags (tag-only).")
    print("Architecture: Embedding (1024) + Pattern (390) -> Tag prediction")
    print("  Pattern features: 2 × 195 tags = 390-dim per-tag n-gram matching")
    print()

    try:
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = UnifiedTagClassifierTrainer()

        # Load data
        df_train, df_eval = trainer.load_data()

        logger.info(f"\n  Dataset summary:")
        logger.info(f"    Training questions: {len(df_train)}")
        logger.info(f"    Eval questions: {len(df_eval)}")
        logger.info(f"    Unique tags: {df_train['tag'].nunique()}")

        # Train unified model with early stopping
        start_time = time.time()
        model = trainer.train(
            df_train,
            epochs=200,      # Max epochs
            batch_size=64,
            lr=0.001,
            patience=15      # Early stopping patience
        )
        train_time = time.time() - start_time

        # Save model
        trainer.save_model(model, models_dir)

        # Summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE - UNIFIED MODEL SUMMARY")
        print("=" * 80)
        print(f"\nModel: unified_tag_classifier.pth")
        print(f"Training time: {train_time:.1f}s ({train_time/60:.1f} minutes)")
        print(f"\nTraining Details:")
        print(f"  Epochs trained: {trainer.trained_epochs}/200")
        print(f"  Best epoch: {trainer.best_epoch}")
        print(f"  Best val accuracy: {trainer.best_val_acc:.2f}%")
        print(f"  Early stopped: {'Yes' if trainer.early_stopped else 'No'}")
        print(f"\nModel saved to: {models_dir / 'unified_tag_classifier.pth'}")
        print(f"\n💡 To evaluate:")
        print(f"   # TODO: add idea/evals scripts")
        print("\n" + "=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
