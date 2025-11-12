"""
FAISS semantic search matcher copied into idea/ so the prototype is self-contained.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import faiss
import numpy as np
import pandas as pd

from ..config import SemanticSearchConfig
from ..utils.model_cache import get_shared_embedding_model

logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


class FaissMatcher:
    """
    Semantic similarity search against FAISS indices.
    """

    def __init__(self, config: SemanticSearchConfig):
        self.config = config
        self.models_dir = Path(config.models_dir)

        self.embedding_model = None
        self.indices = {}
        self.question_mapping = None
        self.tag_to_answer = {}

    def initialize(self):
        logger.info("Initializing FAISS matcher (idea)...")
        logger.info("  Loading embedding model: %s", self.config.embedding_model)
        self.embedding_model = get_shared_embedding_model(self.config.embedding_model)

        self._load_faiss_indices()
        self._load_question_mapping()
        self._load_answers()
        logger.info("  ✓ FAISS matcher ready")

    def _load_faiss_indices(self):
        similarity_dir = self.models_dir
        if not similarity_dir.exists():
            raise FileNotFoundError(
                f"FAISS models directory not found: {similarity_dir}. "
                "Copy indices into idea/models/semantic or update IdeaConfig.semantic.models_dir."
            )

        global_index = similarity_dir / "faiss_index_global.index"
        if not global_index.exists():
            raise FileNotFoundError(f"Global FAISS index missing: {global_index}")

        self.indices['global'] = faiss.read_index(str(global_index))
        logger.info("  Loaded global index with %d vectors", self.indices['global'].ntotal)

    def _load_question_mapping(self):
        mapping_file = self.models_dir / "question_mapping.csv"
        if not mapping_file.exists():
            raise FileNotFoundError(f"question_mapping.csv not found at {mapping_file}")

        self.question_mapping = pd.read_csv(mapping_file)
        logger.info("  Loaded question mapping (%d rows)", len(self.question_mapping))

    def _load_answers(self):
        path = DATASETS_DIR / "tag_to_answer.json"
        if not path.exists():
            raise FileNotFoundError(f"tag_to_answer.json not found at {path}")

        with open(path, 'r', encoding='utf-8') as f:
            self.tag_to_answer = json.load(f)
        logger.info("  Loaded %d answers from %s", len(self.tag_to_answer), path)

    def _encode_query(self, query: str) -> np.ndarray:
        embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=self.config.normalize_embeddings
        )
        return embedding[0].astype('float32').reshape(1, -1)

    def _search_index(
        self,
        query: str,
        faiss_index: faiss.Index,
        result_indices: Optional[List[int]],
        top_k: int,
        source_index_name: str
    ) -> List[Dict[str, Any]]:
        query_embedding = self._encode_query(query)
        similarities, indices = faiss_index.search(query_embedding, top_k)
        similarities = similarities[0]
        indices = indices[0]

        results = []
        for similarity, idx in zip(similarities, indices):
            original_idx = result_indices[idx] if result_indices else idx
            if original_idx >= len(self.question_mapping):
                continue

            row = self.question_mapping.iloc[original_idx]
            results.append({
                'question': row['question'],
                'tag': row['tag'],
                'similarity': float(similarity),
                'score': float(similarity),
                'answer': self.tag_to_answer.get(row['tag'], 'Answer not found'),
                'source_index': source_index_name
            })
        return results

    def search_global(self, query: str, top_k: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start = time.time()
        results = self._search_index(
            query,
            self.indices['global'],
            None,
            top_k,
            source_index_name='global'
        )
        total_time = (time.time() - start) * 1000
        metadata = {
            'strategy_used': 'global',
            'indices_queried': ['global'],
            'num_vectors_searched': self.indices['global'].ntotal,
            'search_time_ms': round(total_time, 2),
            'num_results': len(results)
        }
        return results, metadata
