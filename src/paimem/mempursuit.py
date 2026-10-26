"""Fixed-budget MemPursuit with a fully corrective simplex refit.

Adapted from FengQi-HITSZ/MemPursuit, commit
5138dd0e6f6dbf843692526a44b696a4b9f4bcfb:
  https://github.com/FengQi-HITSZ/MemPursuit/blob/5138dd0e6f6dbf843692526a44b696a4b9f4bcfb/mempursuit/core.py
  https://github.com/FengQi-HITSZ/MemPursuit/blob/5138dd0e6f6dbf843692526a44b696a4b9f4bcfb/mempursuit/simplex.py
Licensed under Apache-2.0; see
third_party/licenses/MemPursuit-Apache-2.0.txt.

Modified for paimem: combined the standalone NumPy selector and
solver, added local configuration and finite-value validation, and omitted the
optional adaptive/Numba paths. Selection order and zero-weight entries are
preserved. This module only consumes existing vectors and performs no model
calls, training, persistence, or input-array mutation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


_NUMERICAL_EPS = 1e-12


@dataclass(frozen=True)
class MemPursuitConfig:
    enabled: bool = False
    candidates: int = 200
    ridge: float = 1e-10

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("mempursuit_enabled must be a boolean")
        if type(self.candidates) is not int or self.candidates <= 0:
            raise ValueError("mempursuit_candidates must be a positive integer")
        _validate_nonnegative_finite(self.ridge, "mempursuit_ridge")

    def settings(self) -> dict[str, bool | int | float]:
        return {f"mempursuit_{name}": value for name, value in asdict(self).items()}


def _validate_nonnegative_finite(value: float, name: str) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class PursuitStep:
    """Diagnostics recorded after one select-and-refit step."""

    candidate_index: int
    selection_score: float
    residual_norm: float
    squared_error: float
    relative_improvement: float
    squared_error_reduction: float
    weights: NDArray[np.float64]


@dataclass(frozen=True)
class MemPursuitResult:
    """Selected candidate indices in entry order and their final refit weights."""

    selected_indices: NDArray[np.int64]
    weights: NDArray[np.float64]
    steps: tuple[PursuitStep, ...]
    stop_reason: Literal["budget"]


def select(
    query_embedding: ArrayLike,
    candidate_embeddings: ArrayLike,
    budget: int,
    *,
    ridge: float = 1e-10,
) -> MemPursuitResult:
    """Select exactly ``budget`` entries, retaining their greedy entry order.

    Inputs are copied to float64 and L2-normalized. Every iteration selects the
    largest residual correlation, then refits all selected weights subject to
    nonnegativity and a unit sum. Zero-weight entries remain in the returned
    sequence; coefficients are diagnostics, not ranking scores. Exact ties use
    the first remaining candidate in the caller's input order.
    """

    query, candidates = _prepare_embeddings(query_embedding, candidate_embeddings, normalize=True)
    candidate_count = candidates.shape[0]
    _validate_options(budget=budget, candidate_count=candidate_count, ridge=ridge)

    # G_ij = <f(m_i), f(m_j)> and b_i = <f(m_i), f(q)> (Appendix A.1).
    gram = np.asarray(candidates @ candidates.T, dtype=np.float64)
    gram = 0.5 * (gram + gram.T)
    linear = np.asarray(candidates @ query, dtype=np.float64)
    query_norm_squared = float(query @ query)

    # Reuse contiguous buffers rather than copying the full Gram matrix at
    # every pursuit step.
    selected_columns = np.empty((budget, candidate_count), dtype=np.float64)
    selected_gram = np.empty((budget, budget), dtype=np.float64)
    selected_linear = np.empty(budget, dtype=np.float64)
    warm_buffer = np.zeros(budget, dtype=np.float64)
    selected: list[int] = []
    weights = np.zeros(0, dtype=np.float64)
    residual_scores = linear.copy()
    unavailable = np.zeros(candidate_count, dtype=np.bool_)
    previous_squared_error = max(query_norm_squared, 0.0)
    previous_regularized_objective: float | None = None
    steps: list[PursuitStep] = []
    stop_reason: Literal["budget"] = "budget"

    for _ in range(budget):
        conditional_scores = residual_scores.copy()
        conditional_scores[unavailable] = -np.inf
        candidate_index = int(np.argmax(conditional_scores))
        if not np.isfinite(conditional_scores[candidate_index]):
            raise RuntimeError("candidate pool was exhausted before reaching the requested budget")

        next_selected = [*selected, candidate_index]
        selected_array = np.asarray(next_selected, dtype=np.int64)
        previous_size = len(selected)
        next_size = previous_size + 1
        selected_columns[previous_size] = gram[:, candidate_index]
        selected_gram[previous_size, :next_size] = gram[candidate_index, selected_array]
        selected_gram[:next_size, previous_size] = selected_gram[previous_size, :next_size]
        selected_linear[previous_size] = linear[candidate_index]
        active_gram = selected_gram[:next_size, :next_size]
        active_linear = selected_linear[:next_size]
        warm_buffer[:previous_size] = weights
        warm_buffer[previous_size] = 0.0
        warm_start = warm_buffer[:next_size] if selected else None
        next_weights = solve_simplex_qp(
            active_gram,
            active_linear,
            ridge=ridge,
            warm_start=warm_start,
        )

        squared_error = (
            query_norm_squared
            - 2.0 * float(next_weights @ active_linear)
            + float(next_weights @ active_gram @ next_weights)
        )
        squared_error = max(squared_error, 0.0)
        regularized_objective = squared_error + ridge * float(next_weights @ next_weights)
        if previous_regularized_objective is not None:
            tolerance = 1e-9 * max(1.0, abs(previous_regularized_objective))
            if regularized_objective > previous_regularized_objective + tolerance:
                raise RuntimeError("fully corrective objective increased beyond numerical tolerance")

        relative_improvement = (previous_squared_error - squared_error) / max(
            previous_squared_error,
            _NUMERICAL_EPS,
        )
        step = PursuitStep(
            candidate_index=candidate_index,
            selection_score=float(conditional_scores[candidate_index]),
            residual_norm=float(np.sqrt(squared_error)),
            squared_error=squared_error,
            relative_improvement=relative_improvement,
            squared_error_reduction=previous_squared_error - squared_error,
            weights=next_weights.copy(),
        )

        selected = next_selected
        weights = next_weights
        unavailable[candidate_index] = True
        steps.append(step)

        # Eq. 9: every candidate is rescored against the updated reconstruction.
        residual_scores = linear - selected_columns[:next_size].T @ weights

        previous_squared_error = squared_error
        previous_regularized_objective = regularized_objective

    return MemPursuitResult(
        selected_indices=np.asarray(selected, dtype=np.int64),
        weights=weights.copy(),
        steps=tuple(steps),
        stop_reason=stop_reason,
    )


def _prepare_embeddings(
    query_embedding: ArrayLike,
    candidate_embeddings: ArrayLike,
    *,
    normalize: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    query = np.asarray(query_embedding, dtype=np.float64)
    candidates = np.asarray(candidate_embeddings, dtype=np.float64)
    if query.ndim != 1 or query.size == 0:
        raise ValueError("query_embedding must be a nonempty one-dimensional vector")
    if candidates.ndim != 2 or candidates.shape[0] == 0:
        raise ValueError("candidate_embeddings must be a nonempty two-dimensional matrix")
    if candidates.shape[1] != query.shape[0]:
        raise ValueError("query and candidate embedding dimensions must match")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(candidates)):
        raise ValueError("embeddings must contain only finite values")

    query_norm = float(np.linalg.norm(query))
    candidate_norms = np.linalg.norm(candidates, axis=1)
    if not np.isfinite(query_norm) or not np.all(np.isfinite(candidate_norms)):
        raise ValueError("embedding norms must be finite")
    if query_norm <= 1e-12 or np.any(candidate_norms <= 1e-12):
        raise ValueError("query and candidate embeddings must be nonzero")
    if normalize:
        query = query / query_norm
        candidates = candidates / candidate_norms[:, None]
    return np.ascontiguousarray(query), np.ascontiguousarray(candidates)


def _validate_options(*, budget: int, candidate_count: int, ridge: float) -> None:
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise TypeError("budget must be an integer")
    if budget < 1 or budget > candidate_count:
        raise ValueError("budget must satisfy 1 <= budget <= number of candidates")
    _validate_nonnegative_finite(ridge, "ridge")


def solve_simplex_qp(
    gram: ArrayLike,
    linear: ArrayLike,
    *,
    ridge: float = 1e-10,
    warm_start: ArrayLike | None = None,
    feasibility_tolerance: float = 1e-10,
) -> NDArray[np.float64]:
    """Solve the fully corrective MemPursuit refit.

    This minimizes

        0.5 * alpha.T @ (gram + ridge * I) @ alpha - linear.T @ alpha

    subject to ``alpha >= 0`` and ``sum(alpha) == 1``. The omitted constant
    is half the squared norm of the query embedding.
    """

    matrix = np.asarray(gram, dtype=np.float64)
    vector = np.asarray(linear, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be a square matrix")
    if vector.shape != (matrix.shape[0],):
        raise ValueError("linear must have shape (gram.shape[0],)")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
        raise ValueError("gram and linear must contain only finite values")
    _validate_nonnegative_finite(ridge, "ridge")
    _validate_nonnegative_finite(feasibility_tolerance, "feasibility_tolerance")

    size = matrix.shape[0]
    if size == 0:
        return np.zeros(0, dtype=np.float64)
    if size == 1:
        return np.ones(1, dtype=np.float64)

    hessian = np.ascontiguousarray(0.5 * (matrix + matrix.T))
    hessian.flat[:: size + 1] += float(ridge)
    tolerance = max(float(feasibility_tolerance), _NUMERICAL_EPS)
    kkt_tolerance = max(10.0 * tolerance, 1e-9)

    solution = _feasible_warm_start(warm_start, size)
    use_warm_start = solution is not None
    if solution is None:
        solution = np.zeros(size, dtype=np.float64)
        vertex_objectives = 0.5 * np.diag(hessian) - vector
        solution[int(np.argmin(vertex_objectives))] = 1.0

    return _solve_numpy(
        hessian,
        vector,
        solution,
        use_warm_start=use_warm_start,
        feasibility_tolerance=tolerance,
        kkt_tolerance=kkt_tolerance,
    )


def _feasible_warm_start(warm_start: ArrayLike | None, size: int) -> NDArray[np.float64] | None:
    if warm_start is None:
        return None
    value = np.asarray(warm_start, dtype=np.float64)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        return None
    value = np.maximum(value, 0.0)
    total = float(value.sum())
    if total <= _NUMERICAL_EPS:
        return None
    return value / total


def _kkt_status(
    hessian: NDArray[np.float64],
    linear: NDArray[np.float64],
    solution: NDArray[np.float64],
    active: NDArray[np.bool_],
    feasibility_tolerance: float,
    kkt_tolerance: float,
) -> tuple[bool, int]:
    gradient = hessian @ solution - linear
    multiplier = -float(np.mean(gradient[active]))
    active_stationary = (
        float(np.max(np.abs(gradient[active] + multiplier))) <= kkt_tolerance
        and abs(float(solution.sum()) - 1.0) <= kkt_tolerance
        and float(np.min(solution)) >= -feasibility_tolerance
    )
    excluded = np.flatnonzero(~active)
    violating = -1
    if excluded.size:
        reduced_gradient = gradient[excluded] + multiplier
        position = int(np.argmin(reduced_gradient))
        if reduced_gradient[position] < -kkt_tolerance:
            violating = int(excluded[position])
    return active_stationary, violating


def _solve_numpy(
    hessian: NDArray[np.float64],
    linear: NDArray[np.float64],
    solution: NDArray[np.float64],
    *,
    use_warm_start: bool,
    feasibility_tolerance: float,
    kkt_tolerance: float,
) -> NDArray[np.float64]:
    size = solution.size
    active = solution > feasibility_tolerance
    if not np.any(active):
        active[int(np.argmax(solution))] = True

    if use_warm_start and np.all(solution[~active] == 0.0):
        stationary, violating = _kkt_status(
            hessian,
            linear,
            solution,
            active,
            feasibility_tolerance,
            kkt_tolerance,
        )
        if stationary:
            if violating < 0:
                return solution
            active[violating] = True

    workspace = (
        np.empty((size + 1) ** 2, dtype=np.float64),
        np.empty(size + 1, dtype=np.float64),
    )
    max_iterations = max(100, 10 * size)
    for _ in range(max_iterations):
        active_indices = np.flatnonzero(active)
        active_solution = _solve_equality_kkt(
            hessian[np.ix_(active_indices, active_indices)],
            linear[active_indices],
            workspace=workspace,
        )

        if np.any(active_solution < -feasibility_tolerance):
            current = solution[active_indices]
            direction = active_solution - current
            decreasing = direction < -feasibility_tolerance
            if not np.any(decreasing):
                raise RuntimeError("simplex working set produced an infeasible direction")
            step_size = min(float(np.min(current[decreasing] / -direction[decreasing])), 1.0)
            next_active = current + step_size * direction
            next_active[np.abs(next_active) <= feasibility_tolerance] = 0.0
            solution.fill(0.0)
            solution[active_indices] = np.maximum(next_active, 0.0)
            _renormalize(solution)
            active = solution > feasibility_tolerance
            continue

        solution.fill(0.0)
        solution[active_indices] = np.maximum(active_solution, 0.0)
        _renormalize(solution)
        active = solution > feasibility_tolerance
        stationary, violating = _kkt_status(
            hessian,
            linear,
            solution,
            active,
            feasibility_tolerance,
            kkt_tolerance,
        )
        if violating >= 0:
            active[violating] = True
            continue
        if stationary:
            return solution

    raise RuntimeError(f"simplex working set did not converge in {max_iterations} iterations")


def _renormalize(solution: NDArray[np.float64]) -> None:
    total = float(solution.sum())
    if total <= _NUMERICAL_EPS:
        raise RuntimeError("simplex working set lost primal feasibility")
    solution /= total


def _solve_equality_kkt(
    hessian: NDArray[np.float64],
    linear: NDArray[np.float64],
    *,
    workspace: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Solve an equality-constrained QP on the current positive support."""

    size = hessian.shape[0]
    kkt = workspace[0][: (size + 1) ** 2].reshape(size + 1, size + 1)
    kkt[:size, :size] = hessian
    kkt[:size, size] = 1.0
    kkt[size, :size] = 1.0
    kkt[size, size] = 0.0
    rhs = workspace[1][: size + 1]
    rhs[:size] = linear
    rhs[size] = 1.0
    try:
        return np.linalg.solve(kkt, rhs)[:-1]
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(kkt, rhs, rcond=None)[0][:-1]

