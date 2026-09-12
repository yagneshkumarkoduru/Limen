"""Inference baselines with calibrated confidence output.

The benchmark requires at least two approaches: a classical feature pipeline
and a temporal baseline. Both expose the same interface so the safety
envelope can consume either without knowing which is upstream - that
decoupling is the point of the confidence-coupled architecture.

- ``LdaBaseline``: per-window Linear Discriminant Analysis; confidence is
  the softmax posterior of the predicted class.
- ``TemporalBaseline``: LDA plus a deterministic sliding-window majority
  vote with hysteresis; confidence is the smoothed posterior mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


@dataclass(frozen=True)
class Prediction:
    """One inference result consumed by the safety envelope."""

    intent: str
    confidence: float  # in [0, 1]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")


class LdaBaseline:
    """Classical feature pipeline: LDA over time-domain features."""

    def __init__(self) -> None:
        self._model = LinearDiscriminantAnalysis()
        self._classes: tuple[str, ...] = ()
        self._fitted = False

    def fit(self, features: NDArray[np.float64], labels: list[str]) -> None:
        if len(labels) == 0:
            raise ValueError("cannot fit LDA on an empty training set")
        self._model.fit(features, labels)
        self._classes = tuple(str(c) for c in self._model.classes_)
        self._fitted = True

    @property
    def classes(self) -> tuple[str, ...]:
        return self._classes

    def predict(self, features: NDArray[np.float64]) -> Prediction:
        if not self._fitted:
            raise RuntimeError("LdaBaseline.predict called before fit")
        proba = self._model.predict_proba(features.reshape(1, -1))[0]
        best = int(np.argmax(proba))
        return Prediction(intent=self._classes[best], confidence=float(proba[best]))


class TemporalBaseline:
    """Temporal baseline: LDA + sliding majority vote with hysteresis.

    A decision changes only when ``hysteresis`` of the last ``history``
    windows agree on a different class. Confidence is the mean posterior of
    the current class over the history window.
    """

    def __init__(self, history: int = 5, hysteresis: int = 4) -> None:
        if hysteresis > history:
            raise ValueError("hysteresis cannot exceed history length")
        self._lda = LdaBaseline()
        self._history = history
        self._hysteresis = hysteresis
        self._recent: list[Prediction] = []
        self._committed: str | None = None

    def fit(self, features: NDArray[np.float64], labels: list[str]) -> None:
        self._lda.fit(features, labels)

    @property
    def classes(self) -> tuple[str, ...]:
        return self._lda.classes

    def reset_stream(self) -> None:
        """Clear temporal state - required between replayed segments."""
        self._recent = []
        self._committed = None

    def predict(self, features: NDArray[np.float64]) -> Prediction:
        raw = self._lda.predict(features)
        self._recent.append(raw)
        if len(self._recent) > self._history:
            self._recent.pop(0)

        votes: dict[str, int] = {}
        for p in self._recent:
            votes[p.intent] = votes.get(p.intent, 0) + 1
        top_intent = max(votes, key=votes.get)

        if self._committed is None:
            self._committed = raw.intent
        elif top_intent != self._committed and votes[top_intent] >= self._hysteresis:
            self._committed = top_intent

        committed_conf = float(
            np.mean([p.confidence for p in self._recent if p.intent == self._committed])
            if any(p.intent == self._committed for p in self._recent)
            else raw.confidence
        )
        return Prediction(intent=self._committed, confidence=committed_conf)
