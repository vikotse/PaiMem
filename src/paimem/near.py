"""NEAR (arXiv:2610.13314), equations 1--4 and Appendix C.

Exact NumPy scoring replaces the paper's ANN index. Only cached embeddings are
used; no query labels or text rewriting. Construction is quadratic in bank size
but pairwise similarities are blocked to avoid an N-by-N allocation.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import math
import threading
import time

import numpy as np


@dataclass(frozen=True)
class NearConfig:
    enabled: bool = False
    layers: int = 1
    weight: float = 1.0
    normalize: bool = False
    neighbors: int = 50
    anchors: int = 16
    routing: int = 4
    beta: float = 10.0
    cache_max_users: int = 4
    cache_max_mb: float = 256.0

    def __post_init__(self):
        for name in ("enabled", "normalize"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"near_{name} must be a boolean")
        for name in ("layers", "neighbors", "anchors", "routing", "cache_max_users"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"near_{name} must be a positive integer")
        for name in ("weight", "beta", "cache_max_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"near_{name} must be finite")
            if value < 0 or (name != "weight" and value == 0):
                raise ValueError(f"near_{name} is outside its valid range")

    def settings(self):
        return {f"near_{name}": value for name, value in asdict(self).items()}


def unit(vectors):
    """Numerically zero residuals stay zero instead of amplifying roundoff."""
    values = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-7)


def score_scale(vectors):
    """sqrt(mean_i Var_j[z_i dot z_j]), including the diagonal (paper C)."""
    values = np.asarray(vectors, dtype=np.float64)
    if len(values) == 0:
        return 1e-6
    # E[zz^T] and Cov(z) give the same statistic without all pairwise scores.
    second = values.T @ values / len(values)
    mean = values.mean(axis=0)
    covariance = second - np.outer(mean, mean)
    return max(1e-6, math.sqrt(max(0.0, float(np.sum(second * covariance)))))


def neighborhood_references(states, neighbors):
    count, dimension = states.shape
    references = np.zeros((count, dimension), dtype=np.float64)
    if count < 2:
        return references
    width = min(neighbors, count - 1)
    # At most about 16 MB of similarity scores per block.
    block = max(1, min(256, 2_000_000 // count))
    for start in range(0, count, block):
        end = min(count, start + block)
        scores = states[start:end] @ states.T
        scores[np.arange(end - start), np.arange(start, end)] = -np.inf
        positions = np.argpartition(-scores, width - 1, axis=1)[:, :width]
        references[start:end] = unit(states[positions].mean(axis=1))
    return references


def reference_table(states, references, count):
    """Deterministic spherical k-means, farthest-first seed 42, <=50 steps.

    Empty clusters are discarded. Duplicate/zero states can yield fewer anchors.
    The paper specifies spherical k-means but not its initialization/stopping rule.
    """
    usable = states[np.linalg.norm(states, axis=1) > 1e-7]
    if len(usable) == 0:
        return np.zeros((1, states.shape[1])), np.zeros((1, states.shape[1]))
    first = int(np.random.default_rng(42).integers(len(usable)))
    centers = [usable[first]]
    closest = usable @ centers[0]
    for _ in range(1, min(count, len(usable))):
        index = int(np.argmin(closest))
        if closest[index] >= 1 - 1e-10:
            break
        centers.append(usable[index])
        closest = np.maximum(closest, usable @ centers[-1])
    anchors = np.asarray(centers)
    for _ in range(50):
        labels = np.argmax(states @ anchors.T, axis=1)
        means = [states[labels == i].mean(axis=0)
                 for i in range(len(anchors)) if np.any(labels == i)]
        updated = unit(means)
        if updated.shape == anchors.shape and np.allclose(updated, anchors, atol=1e-7, rtol=0):
            anchors = updated
            break
        anchors = updated
    labels = np.argmax(states @ anchors.T, axis=1)
    active = [i for i in range(len(anchors)) if np.any(labels == i)]
    directions = unit([references[labels == i].mean(axis=0) for i in active])
    return anchors[active], directions


class NearBank:
    def __init__(self, matrix, config: NearConfig):
        started = time.perf_counter()
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("NEAR requires a finite memory matrix")
        if len(values) and np.any(np.linalg.norm(values, axis=1) <= 1e-7):
            raise ValueError("NEAR requires nonzero memory embeddings")
        self.config = config
        self.views = [unit(values)]
        self.tables = []
        self.scales = [score_scale(self.views[0])]
        state = self.views[0]
        # Singleton/empty banks have no neighborhood: preserve base matching.
        for _ in range(config.layers if len(values) >= 2 else 0):
            reference = neighborhood_references(state, config.neighbors)
            self.tables.append(reference_table(state, reference, config.anchors))
            residual = state - np.sum(state * reference, axis=1, keepdims=True) * reference
            residual[np.linalg.norm(residual, axis=1) <= 1e-7] = 0
            state = unit(residual)
            view = state if config.normalize else residual
            self.views.append(view)
            self.scales.append(score_scale(view))
        self.nbytes = sum(v.nbytes for v in self.views) + sum(a.nbytes + v.nbytes for a, v in self.tables)
        self.build_seconds = time.perf_counter() - started

    def score(self, queries):
        values = np.asarray(queries, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != self.views[0].shape[1]
                or not np.all(np.isfinite(values)) or np.any(np.linalg.norm(values, axis=1) <= 1e-7)):
            raise ValueError("NEAR requires finite nonzero query embeddings of the bank dimension")
        state = unit(values)
        scores = self.views[0] @ state.T
        if not self.tables:
            return scores
        scores = scores / self.scales[0]
        for layer, (anchors, directions) in enumerate(self.tables, 1):
            similarity = state @ anchors.T
            positions = np.argsort(-similarity, axis=1, kind="stable")[:, :self.config.routing]
            logits = np.take_along_axis(similarity, positions, axis=1) * self.config.beta
            weights = np.exp(logits - logits.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)
            reference = unit(np.sum(directions[positions] * weights[:, :, None], axis=1))
            residual = state - np.sum(state * reference, axis=1, keepdims=True) * reference
            residual[np.linalg.norm(residual, axis=1) <= 1e-7] = 0
            state = unit(residual)
            view = state if self.config.normalize else residual
            scores += (self.config.weight / self.config.layers / self.scales[layer]) * (self.views[layer] @ view.T)
        return scores


class NearCache:
    """Bounded process-local cache; every key includes exact IDs and vectors.

    Builds are serialized to bound simultaneous construction memory. A search
    owns its immutable bank snapshot even if Add/Clear invalidates the cache.
    """
    def __init__(self, config):
        self.config = config
        self._entries = OrderedDict()
        self._lock = threading.RLock()

    def get(self, user_id, ids, matrix):
        digest = hashlib.sha256()
        for identity in ids:
            encoded = str(identity).encode()
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        digest.update(np.ascontiguousarray(matrix).tobytes())
        signature = digest.digest()
        with self._lock:
            found = self._entries.get(user_id)
            if found is not None and found[0] == signature:
                self._entries.move_to_end(user_id)
                return found[1], True
            bank = NearBank(matrix, self.config)
            self._entries.pop(user_id, None)
            if bank.nbytes <= self.config.cache_max_mb * 1024**2:
                self._entries[user_id] = (signature, bank)
                while (len(self._entries) > self.config.cache_max_users or
                       sum(value[1].nbytes for value in self._entries.values()) > self.config.cache_max_mb * 1024**2):
                    self._entries.popitem(last=False)
            return bank, False

    def invalidate(self, user_id=None):
        with self._lock:
            if user_id is None:
                self._entries.clear()
            else:
                self._entries.pop(user_id, None)
