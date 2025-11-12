"""
Classifier matcher dedicated to the idea pipeline.

Forked from components.sts.inference.classifier_matcher to keep the experimental
codebase self-contained.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.model_cache import get_shared_embedding_model

from ..config import TagClassifierConfig
from .unified_tag_classifier import UnifiedTagClassifier

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
FEATURES_DIR = DATASETS_DIR / "features"


class TagClassifierMatcher:
    """Predicts tags using the UnifiedTagClassifier."""

    def __init__(self, config: TagClassifierConfig):
        self.config = config
        self.models_dir = Path(config.models_dir)
        self.embedding_model_name = config.embedding_model

        self.device = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.model: Optional[UnifiedTagClassifier] = None

        self.tag_encoder = None
        self.pattern_mean = None
        self.pattern_std = None
        self.tag_patterns = None
        self.tags_sorted = None

        self.tag_to_answer: Dict[str, str] = {}

    def initialize(self):
        logger.info("Loading tag classifier matcher from %s", self.models_dir)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        logger.info("  Device: %s", self.device)

        logger.info("  Loading embedding model: %s", self.embedding_model_name)
        self.embedding_model = get_shared_embedding_model(self.embedding_model_name)

        model_file = self.models_dir / "unified_tag_classifier.pth"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Classifier model not found at {model_file}. "
                "Run idea/training/train_tag_classifier.py first."
            )

        checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        num_tags = checkpoint['num_tags']
        self.tag_encoder = checkpoint['tag_encoder_classes']
        self.pattern_mean = np.array(checkpoint['pattern_mean'], dtype=np.float32).squeeze()
        self.pattern_std = np.array(checkpoint['pattern_std'], dtype=np.float32).squeeze()

        if self.pattern_mean.ndim != 1:
            self.pattern_mean = self.pattern_mean.reshape(-1)
        if self.pattern_std.ndim != 1:
            self.pattern_std = self.pattern_std.reshape(-1)

        zero_std = self.pattern_std == 0
        if np.any(zero_std):
            logger.warning(
                "Pattern std contained %d zeros; replacing with 1.0",
                int(np.sum(zero_std))
            )
            self.pattern_std[zero_std] = 1.0

        logger.info("  Tags: %d", num_tags)
        self._load_ngram_patterns(self.config.ngram_mode)

        pattern_dim = len(self.tags_sorted) * 2
        logger.info("  Pattern dim: %d", pattern_dim)

        self.model = UnifiedTagClassifier(
            embedding_dim=self.embedding_model.get_sentence_embedding_dimension(),
            pattern_dim=pattern_dim,
            num_tags=num_tags,
            dropout=0.5
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self._load_answers()
        logger.info("  ✓ Tag classifier matcher ready")

    def _load_answers(self):
        path_options = [
            DATASETS_DIR / "tag_to_answer.json",
            self.models_dir.parent / "datasets" / "tag_to_answer.json"
        ]
        for path in path_options:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    self.tag_to_answer = json.load(f)
                logger.info("  Loaded %d answers from %s", len(self.tag_to_answer), path)
                return
        raise FileNotFoundError("tag_to_answer.json not found in idea datasets.")

    def _load_ngram_patterns(self, mode: str):
        path = FEATURES_DIR / f"{mode}_ngrams.json"
        if not path.exists():
            raise FileNotFoundError(
                f"N-gram file not found: {path}. Run idea/featurizer/generate_features.py."
            )

        logger.info("  Loading n-gram features from %s", path)
        with open(path, 'r', encoding='utf-8') as f:
            features = json.load(f)

        self.tag_patterns = {}
        for tag_name, tag_data in features["tags"].items():
            self.tag_patterns[tag_name] = {
                "trigrams": set(item["ngram"] for item in tag_data["trigrams"]),
                "fourgrams": set(item["ngram"] for item in tag_data["fourgrams"])
            }

        self.tags_sorted = sorted(self.tag_patterns.keys())
        logger.info("    Loaded features for %d tags", len(self.tags_sorted))

    @staticmethod
    def _extract_ngram_words(text: str, n: int) -> set:
        words = text.split()
        if len(words) < n:
            return set()
        return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}

    def _compute_pattern_features(self, question: str) -> np.ndarray:
        q_trigrams = self._extract_ngram_words(question, 3)
        q_fourgrams = self._extract_ngram_words(question, 4)

        feat = []
        for tag in self.tags_sorted:
            trigram_matches = len(q_trigrams & self.tag_patterns[tag]["trigrams"])
            fourgram_matches = len(q_fourgrams & self.tag_patterns[tag]["fourgrams"])
            feat.extend([trigram_matches, fourgram_matches])
        return np.array(feat, dtype=np.float32)

    def predict(self, question: str, top_k: int = 3) -> Dict[str, any]:
        embedding = self.embedding_model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]

        pattern = self._compute_pattern_features(question)
        pattern = (pattern - self.pattern_mean) / self.pattern_std

        emb_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
        pat_tensor = torch.FloatTensor(pattern).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(emb_tensor, pat_tensor)
            probs = F.softmax(logits, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            top_probs = top_probs.cpu().numpy()
            top_indices = top_indices.cpu().numpy()

        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            tag = self.tag_encoder[idx]
            predictions.append({
                'tag': tag,
                'confidence': float(prob),
                'answer': self.tag_to_answer.get(tag, "Answer not found")
            })

        return {
            'results': predictions,
            'metadata': {
                'num_candidates': len(predictions)
            }
        }

    def search(self, query: str, top_k: int) -> Dict[str, any]:
        start = time.time()
        response = self.predict(query, top_k)
        response['metadata']['inference_time_ms'] = round((time.time() - start) * 1000, 2)
        return response
