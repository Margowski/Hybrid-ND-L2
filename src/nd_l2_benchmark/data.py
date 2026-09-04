"""Synthetic, auditable sequence tasks for state persistence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


N_TYPES = 6
N_KEYS = 6
N_VALUES = 8
N_CONTEXTS = 4
N_RELATIONS = 4

TYPE_WRITE = 0
TYPE_CONTEXT = 1
TYPE_AFFECT = 2
TYPE_RELATION = 3
TYPE_DISTRACTOR = 4
TYPE_QUERY = 5

TYPE_SLICE = slice(0, N_TYPES)
KEY_SLICE = slice(TYPE_SLICE.stop, TYPE_SLICE.stop + N_KEYS)
VALUE_SLICE = slice(KEY_SLICE.stop, KEY_SLICE.stop + N_VALUES)
CONTEXT_SLICE = slice(VALUE_SLICE.stop, VALUE_SLICE.stop + N_CONTEXTS)
RELATION_SLICE = slice(CONTEXT_SLICE.stop, CONTEXT_SLICE.stop + N_RELATIONS)
AFFECT_INDEX = RELATION_SLICE.stop
INPUT_DIM = AFFECT_INDEX + 1


@dataclass(frozen=True)
class BenchmarkData:
    x: np.ndarray
    memory: np.ndarray
    context: np.ndarray
    self_state: np.ndarray
    relation: np.ndarray

    @property
    def targets(self) -> dict[str, np.ndarray]:
        return {
            "memory": self.memory,
            "context": self.context,
            "self_state": self.self_state,
            "relation": self.relation,
        }


def _token(kind: int) -> np.ndarray:
    x = np.zeros(INPUT_DIM, dtype=np.float64)
    x[kind] = 1.0
    return x


def generate_dataset(
    n_samples: int,
    *,
    seq_len: int = 36,
    seed: int = 0,
) -> BenchmarkData:
    """Generate balanced multi-state sequences with independently known labels."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if seq_len < 24:
        raise ValueError("seq_len must be at least 24")

    rng = np.random.default_rng(seed)
    xs = np.zeros((n_samples, seq_len, INPUT_DIM), dtype=np.float64)
    y_mem = np.empty(n_samples, dtype=np.int64)
    y_ctx = np.empty(n_samples, dtype=np.int64)
    y_self = np.empty(n_samples, dtype=np.int64)
    y_rel = np.empty(n_samples, dtype=np.int64)

    for n in range(n_samples):
        # Small continuous distractors make perturbation tests meaningful.
        xs[n, :, :] = rng.normal(0.0, 0.015, size=(seq_len, INPUT_DIM))
        target_key = int(rng.integers(N_KEYS))
        target_value = int(rng.integers(N_VALUES))
        target_relation = int(rng.integers(N_RELATIONS))

        occupied: set[int] = {seq_len - 1}
        write_pos = int(rng.integers(1, 5))
        relation_pos = int(rng.integers(6, 11))
        occupied.update({write_pos, relation_pos})

        tok = _token(TYPE_WRITE)
        tok[KEY_SLICE.start + target_key] = 1.0
        tok[VALUE_SLICE.start + target_value] = 1.0
        xs[n, write_pos] += tok

        tok = _token(TYPE_RELATION)
        tok[KEY_SLICE.start + target_key] = 1.0
        tok[RELATION_SLICE.start + target_relation] = 1.0
        xs[n, relation_pos] += tok

        # Three context changes; the latest one is the target.
        context_candidates = np.array(
            [p for p in range(4, seq_len - 2) if p not in occupied]
        )
        context_positions = sorted(
            rng.choice(context_candidates, size=3, replace=False).tolist()
        )
        current_context = 0
        for pos in context_positions:
            current_context = int(rng.integers(N_CONTEXTS))
            tok = _token(TYPE_CONTEXT)
            tok[CONTEXT_SLICE.start + current_context] = 1.0
            xs[n, pos] += tok
            occupied.add(pos)

        # Signed events define a latent self state by temporal integration.
        affect_sum = 0.0
        candidates = np.array([p for p in range(3, seq_len - 1) if p not in occupied])
        affect_positions = rng.choice(candidates, size=6, replace=False)
        for pos in affect_positions:
            affect = float(rng.choice([-1.0, -0.5, 0.5, 1.0]))
            tok = _token(TYPE_AFFECT)
            tok[AFFECT_INDEX] = affect
            xs[n, pos] += tok
            affect_sum += affect
            occupied.add(int(pos))

        # Keyed distractor facts/relations make simple pooling insufficient.
        other_keys = [k for k in range(N_KEYS) if k != target_key]
        free = [p for p in range(2, seq_len - 1) if p not in occupied]
        rng.shuffle(free)
        for idx, key in enumerate(other_keys[:3]):
            pos = free.pop()
            tok = _token(TYPE_WRITE)
            tok[KEY_SLICE.start + key] = 1.0
            tok[VALUE_SLICE.start + int(rng.integers(N_VALUES))] = 1.0
            xs[n, pos] += tok
            pos = free.pop()
            tok = _token(TYPE_RELATION)
            tok[KEY_SLICE.start + key] = 1.0
            tok[RELATION_SLICE.start + int(rng.integers(N_RELATIONS))] = 1.0
            xs[n, pos] += tok

        for pos in free[:4]:
            xs[n, pos] += _token(TYPE_DISTRACTOR)

        query = _token(TYPE_QUERY)
        query[KEY_SLICE.start + target_key] = 1.0
        xs[n, -1] += query

        y_mem[n] = target_value
        y_ctx[n] = current_context
        y_self[n] = 0 if affect_sum < -0.75 else (2 if affect_sum > 0.75 else 1)
        y_rel[n] = target_relation

    return BenchmarkData(xs, y_mem, y_ctx, y_self, y_rel)


def perturb(data: BenchmarkData, *, sigma: float, seed: int) -> BenchmarkData:
    """Return a copy with additive sensor/input noise and identical targets."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    rng = np.random.default_rng(seed)
    noisy = data.x + rng.normal(0.0, sigma, size=data.x.shape)
    return BenchmarkData(
        noisy,
        data.memory.copy(),
        data.context.copy(),
        data.self_state.copy(),
        data.relation.copy(),
    )
