"""
Experimental dual-leg pipeline that runs STS similarity + 195-tag classification
in tandem and relies on a ranker/decider to pick the best answer.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from .config import IdeaConfig
from .semantic_similarity import SemanticSearchEngine
from .tag_classifier import TagClassifier
from .ranker import DualSignalRanker, RankerConfig as RankerRuntimeConfig

logger = logging.getLogger(__name__)


class IdeaPipeline:
    """
    New pipeline requested by the user – lives entirely under ``idea/`` so
    legacy components remain untouched.
    """

    def __init__(self, config: Optional[IdeaConfig | Dict[str, Any]] = None):
        if isinstance(config, IdeaConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = IdeaConfig.from_dict(config)
        else:
            self.config = IdeaConfig()

        self.semantic = SemanticSearchEngine(self.config.semantic)
        self.classifier = TagClassifier(self.config.classifier)
        self.ranker = DualSignalRanker(
            RankerRuntimeConfig(
                weights=self.config.ranker.weights,
                min_score=self.config.ranker.min_score
            )
        )

        self._initialized = False

    def initialize(self):
        if self._initialized:
            logger.info("IdeaPipeline already initialized")
            return

        logger.info("Initializing IdeaPipeline (dual STS + 195-tag classifier)...")
        self.semantic.initialize()
        self.classifier.initialize()
        self._initialized = True
        logger.info("✓ IdeaPipeline ready")

    def run(
        self,
        question: str,
        fusion_top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        fusion_top_k = fusion_top_k or self.config.fusion_top_k
        start_time = time.time()

        # Leg 1 – STS similarity (global)
        sts_state = self.semantic.search(question, top_k=self.config.semantic.top_k)

        # Leg 2 – Tag classifier (195 tags)
        clf_state = self.classifier.predict(
            question,
            top_k=self.config.classifier.top_k
        )

        # Ranker – fuse signals
        ranked, telemetry = self.ranker.rank(
            sts_state.results,
            clf_state.predictions,
            top_k=fusion_top_k
        )

        best = ranked[0] if ranked else None
        primary_answer = (
            (best or {}).get("answer")
            or (clf_state.predictions[0]["answer"] if clf_state.predictions else None)
            or (sts_state.results[0]["answer"] if sts_state.results else None)
            or "Answer not found"
        )

        total_time = (time.time() - start_time) * 1000

        return {
            "question": question,
            "answer": primary_answer,
            "primary_tag": (best or {}).get("tag"),
            "candidates": ranked,
            "signals": {
                "sts": {
                    "results": sts_state.results,
                    "metadata": sts_state.metadata
                },
                "classifier": {
                    "results": clf_state.predictions,
                    "metadata": clf_state.metadata
                }
            },
            "telemetry": {
                "ranker": telemetry.__dict__,
                "latency_ms": round(total_time, 2),
                "fusion_top_k": fusion_top_k
            }
        }
