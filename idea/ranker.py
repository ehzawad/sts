"""
Signal fusion and reranking logic for the experimental dual-leg pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math


def _score_from_sts(candidate: Dict[str, Any]) -> float:
    if not candidate:
        return 0.0
    if candidate.get("score") is not None:
        return float(candidate["score"])
    if candidate.get("similarity") is not None:
        return float(candidate["similarity"])
    return 0.0


@dataclass
class RankerTelemetry:
    """Structured summary of what the ranker observed."""

    weights: Dict[str, float]
    min_score: float
    num_fused_tags: int
    dropped_below_threshold: int
    input_counts: Dict[str, int]


@dataclass
class RankerConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {"sts": 0.6, "classifier": 0.4})
    min_score: float = 0.05


class DualSignalRanker:
    """
    Blend STS similarity scores with classifier probabilities and surface the best tag.
    """

    def __init__(self, config: Optional[RankerConfig] = None):
        self.config = config or RankerConfig()
        self.weights = self._normalize(self.config.weights)

    @staticmethod
    def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(0.0, w) for w in weights.values())
        if total == 0:
            return {"sts": 0.5, "classifier": 0.5}
        return {key: max(0.0, value) / total for key, value in weights.items()}

    @staticmethod
    def _normalize_scores(values: List[float]) -> List[float]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if math.isclose(lo, hi):
            return [1.0 if val > 0 else 0.0 for val in values]
        denom = hi - lo
        return [(val - lo) / denom for val in values]

    def _fuse(self, sts_results: List[Dict[str, Any]], clf_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        fused: Dict[str, Dict[str, Any]] = {}
        normalized_sts = self._normalize_scores([_score_from_sts(item) for item in sts_results])

        for idx, candidate in enumerate(sts_results):
            tag = candidate.get("tag")
            if not tag:
                continue
            node = fused.setdefault(tag, {"tag": tag, "answer": candidate.get("answer")})
            best = node.get("sts")
            normalized_score = normalized_sts[idx] if idx < len(normalized_sts) else 0.0
            if best is None or normalized_score > best.get("normalized_score", -1):
                node["sts"] = {
                    "raw_score": _score_from_sts(candidate),
                    "normalized_score": normalized_score,
                    "rank": idx,
                    "source_index": candidate.get("source_index"),
                    "question": candidate.get("question"),
                    "similarity": candidate.get("similarity")
                }
            node.setdefault("answer", candidate.get("answer"))

        for idx, candidate in enumerate(clf_results):
            tag = candidate.get("tag")
            if not tag:
                continue
            node = fused.setdefault(tag, {"tag": tag, "answer": candidate.get("answer")})
            best = node.get("classifier")
            confidence = float(candidate.get("confidence", 0.0))
            if best is None or confidence > best.get("confidence", -1):
                node["classifier"] = {
                    "confidence": confidence,
                    "rank": idx,
                    "source": candidate.get("source", "classifier")
                }
            node.setdefault("answer", candidate.get("answer"))

        return fused

    def _score_entry(self, entry: Dict[str, Any]) -> Tuple[float, List[Dict[str, float]], str]:
        contributions: List[Dict[str, float]] = []
        score_sum = 0.0
        total_weight = 0.0
        present: List[str] = []

        if entry.get("sts"):
            weight = self.weights.get("sts", 0.0)
            value = entry["sts"].get("normalized_score", 0.0)
            contributions.append({"signal": "sts", "weight": weight, "value": value})
            score_sum += weight * value
            total_weight += weight
            present.append("sts")

        if entry.get("classifier"):
            weight = self.weights.get("classifier", 0.0)
            value = entry["classifier"].get("confidence", 0.0)
            contributions.append({"signal": "classifier", "weight": weight, "value": value})
            score_sum += weight * value
            total_weight += weight
            present.append("classifier")

        if total_weight == 0:
            return 0.0, contributions, "none"

        final_score = score_sum / total_weight
        source = "+".join(present) if present else "none"
        return final_score, contributions, source

    def rank(
        self,
        sts_results: Optional[List[Dict[str, Any]]],
        classifier_results: Optional[List[Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], RankerTelemetry]:
        sts_results = sts_results or []
        classifier_results = classifier_results or []

        fused_entries = self._fuse(sts_results, classifier_results)
        ranked: List[Dict[str, Any]] = []
        dropped = 0

        for entry in fused_entries.values():
            final_score, contributions, source = self._score_entry(entry)
            node = {
                "tag": entry["tag"],
                "answer": entry.get("answer"),
                "final_score": final_score,
                "contributions": contributions,
                "source": source,
                "signals": {
                    "sts": entry.get("sts"),
                    "classifier": entry.get("classifier")
                }
            }
            if final_score < self.config.min_score:
                dropped += 1
                continue
            ranked.append(node)

        ranked.sort(key=lambda item: item["final_score"], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        telemetry = RankerTelemetry(
            weights=self.weights,
            min_score=self.config.min_score,
            num_fused_tags=len(fused_entries),
            dropped_below_threshold=dropped,
            input_counts={"sts": len(sts_results), "classifier": len(classifier_results)}
        )

        return ranked, telemetry
