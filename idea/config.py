"""
Configuration objects for the experimental dual-signal pipeline that lives
inside the ``idea/`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


def _as_path(value: Optional[str | Path], default: Path) -> Path:
    if value is None:
        return default
    return Path(value)


@dataclass
class SemanticSearchConfig:
    """Settings for the FAISS/STS semantic search leg."""

    models_dir: Path = Path("idea/models/semantic")
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    top_k: int = 20
    strategy: str = "global"  # Always run global for the tandem workflow
    normalize_embeddings: bool = True

    def __post_init__(self):
        self.models_dir = _as_path(self.models_dir, Path("idea/models/semantic"))


@dataclass
class TagClassifierConfig:
    """
    Settings for the 195-tag classifier leg.

    By default we reuse the UnifiedTagClassifier artifacts produced by the
    legacy STS classifier training recipe.
    """

    models_dir: Path = Path("idea/models/tag_classifier")
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    top_k: int = 10
    ngram_mode: str = "manual"  # manual / auto features for pattern branch

    def __post_init__(self):
        self.models_dir = _as_path(self.models_dir, Path("idea/models/tag_classifier"))


@dataclass
class RankerConfig:
    """Weights + filtering for the final decider."""

    weights: Dict[str, float] = field(default_factory=lambda: {"sts": 0.75, "classifier": 0.25})
    min_score: float = 0.05


@dataclass
class IdeaConfig:
    """
    Aggregate config consumed by the experimental pipeline.

    Attributes:
        semantic: SemanticSearchConfig
        classifier: TagClassifierConfig
        ranker: RankerConfig
        fusion_top_k: How many fused answers to keep before returning top-1
        log_dir: Optional location for structured logs/metrics
    """

    semantic: SemanticSearchConfig = field(default_factory=SemanticSearchConfig)
    classifier: TagClassifierConfig = field(default_factory=TagClassifierConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    fusion_top_k: int = 5
    log_dir: Optional[Path] = None

    def __post_init__(self):
        if self.log_dir is not None:
            self.log_dir = _as_path(self.log_dir, Path("idea/logs"))

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "IdeaConfig":
        if not data:
            return cls()

        semantic = data.get("semantic")
        classifier = data.get("classifier")
        ranker = data.get("ranker")

        return cls(
            semantic=SemanticSearchConfig(**semantic) if semantic else SemanticSearchConfig(),
            classifier=TagClassifierConfig(**classifier) if classifier else TagClassifierConfig(),
            ranker=RankerConfig(**ranker) if ranker else RankerConfig(),
            fusion_top_k=data.get("fusion_top_k", cls.fusion_top_k),
            log_dir=data.get("log_dir")
        )
