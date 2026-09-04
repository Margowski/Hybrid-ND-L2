"""Equal-width fixed encoders for the ND-L2 pilot."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .data import (
    AFFECT_INDEX,
    CONTEXT_SLICE,
    INPUT_DIM,
    KEY_SLICE,
    N_CONTEXTS,
    N_KEYS,
    N_RELATIONS,
    N_VALUES,
    RELATION_SLICE,
    TYPE_AFFECT,
    TYPE_CONTEXT,
    TYPE_DISTRACTOR,
    TYPE_QUERY,
    TYPE_RELATION,
    TYPE_SLICE,
    TYPE_WRITE,
    VALUE_SLICE,
)
from .jit_backend import NUMBA_AVAILABLE, run_virtual_recurrence


FEATURE_DIM = 192


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(z, -60.0, 60.0))
    return e / np.sum(e, axis=axis, keepdims=True)


def _fit_size(v: np.ndarray, size: int = FEATURE_DIM) -> np.ndarray:
    flat = np.asarray(v, dtype=np.float64).reshape(-1)
    if flat.size == size:
        return flat
    if flat.size > size:
        return flat[:size]
    return np.pad(flat, (0, size - flat.size))


class Encoder(ABC):
    feature_dim = FEATURE_DIM
    trainable_internal_parameters = 0

    def fit(
        self,
        sequences: np.ndarray,
        targets: dict[str, np.ndarray] | None = None,
    ) -> "Encoder":
        """Fit optional internal parameters; fixed encoders are no-ops."""
        if sequences.ndim != 3 or sequences.shape[-1] != INPUT_DIM:
            raise ValueError(f"expected [batch, time, {INPUT_DIM}] input")
        return self

    @abstractmethod
    def encode_one(self, sequence: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def encode(self, sequences: np.ndarray) -> np.ndarray:
        if sequences.ndim != 3 or sequences.shape[-1] != INPUT_DIM:
            raise ValueError(f"expected [batch, time, {INPUT_DIM}] input")
        return np.vstack([self.encode_one(seq) for seq in sequences])


class CausalAttentionBaseline(Encoder):
    """Transparent four-head causal-attention feature baseline.

    Heads attend to keyed facts, keyed relations, recent context, and affect.
    The encoder is fixed; only the common readout used by the benchmark trains.
    """

    def encode_one(self, sequence: np.ndarray) -> np.ndarray:
        t = sequence.shape[0]
        kinds = sequence[:, TYPE_SLICE]
        query_key = sequence[-1, KEY_SLICE]
        key_match = sequence[:, KEY_SLICE] @ query_key
        recency = np.linspace(-1.0, 1.0, t)

        scores = np.stack(
            [
                6.0 * key_match + 3.5 * kinds[:, TYPE_WRITE],
                6.0 * key_match + 3.5 * kinds[:, TYPE_RELATION],
                4.5 * kinds[:, TYPE_CONTEXT] + 2.0 * recency,
                4.0 * kinds[:, TYPE_AFFECT],
            ],
            axis=0,
        )
        weights = _softmax(scores, axis=1)
        heads = weights @ sequence
        mean = np.mean(sequence, axis=0)
        std = np.std(sequence, axis=0)
        # 4*29 + 2*29 = 174; retain query/type diagnostics to reach 192.
        diagnostics = np.concatenate(
            [
                sequence[-1, TYPE_SLICE],
                np.max(kinds, axis=0),
                np.array([
                    np.max(key_match),
                    np.mean(key_match),
                    np.sum(sequence[:, AFFECT_INDEX]),
                    float(t),
                    1.0,
                    np.linalg.norm(sequence),
                ]),
            ]
        )
        return _fit_size(np.concatenate([heads.ravel(), mean, std, diagnostics]))


class NDStateModel(Encoder):
    """Explicit direct-sum state without multidimensional field feedback."""

    def encode_one(self, sequence: np.ndarray) -> np.ndarray:
        memory = np.zeros((N_KEYS, N_VALUES))
        relation = np.zeros((N_KEYS, N_RELATIONS))
        context = np.zeros((4, N_CONTEXTS))
        affect = np.zeros(24)
        self_state = np.zeros(24)
        semantic = np.zeros(36)

        for step, x in enumerate(sequence):
            kind = x[TYPE_SLICE]
            key = x[KEY_SLICE]
            value = x[VALUE_SLICE]
            rel = x[RELATION_SLICE]
            ctx = x[CONTEXT_SLICE]
            a = float(x[AFFECT_INDEX])

            memory *= 0.995
            relation *= 0.995
            memory += np.outer(key, value) * kind[TYPE_WRITE]
            relation += np.outer(key, rel) * kind[TYPE_RELATION]

            context *= 0.94
            context[0] += ctx * kind[TYPE_CONTEXT]
            context[1] += context[0] * 0.12
            context[2] = np.tanh(context[2] + context[1] * 0.05)
            context[3] = np.maximum(context[3] * 0.9, np.abs(context[0]))

            affect *= 0.975
            affect[0] += a * kind[TYPE_AFFECT]
            affect[1:5] += ctx * a * 0.08
            affect[5:11] += key * a * 0.04
            affect[11:] = np.tanh(affect[11:] + np.mean(x) * 0.03)

            self_state *= 0.998
            self_state[0] += a * kind[TYPE_AFFECT]
            self_state[1] += np.sign(a) * abs(a) ** 0.5 * kind[TYPE_AFFECT]
            self_state[2] = np.tanh(self_state[2] + affect[0] * 0.04)
            self_state[3:7] += context[0] * 0.01
            self_state[7:13] += key * kind[TYPE_QUERY] * 0.04
            self_state[13:] = np.tanh(self_state[13:] + np.mean(semantic) * 0.02)

            raw = np.concatenate([kind, key, value, ctx, rel, [a, step / len(sequence)]])
            semantic *= 0.97
            semantic[: raw.size] += raw * 0.08
            semantic[raw.size :] += np.mean(raw) * 0.02

        query_key = sequence[-1, KEY_SLICE]
        retrieved = np.concatenate([query_key @ memory, query_key @ relation])
        features = np.concatenate(
            [memory.ravel(), relation.ravel(), context.ravel(), affect,
             self_state, semantic, retrieved]
        )
        return _fit_size(features)


def _diffuse(field: np.ndarray, rate: float) -> np.ndarray:
    """One stable Neumann-like diffusion step over the basis/mode axis."""
    if field.shape[0] < 2:
        return field
    left = np.vstack([field[:1], field[:-1]])
    right = np.vstack([field[1:], field[-1:]])
    return field + rate * (left - 2.0 * field + right)


def _diffuse_batch(field: np.ndarray, rate: float) -> np.ndarray:
    """Apply the same diffusion as :func:`_diffuse` to ``[batch, mode, channel]``."""
    if field.ndim != 3:
        raise ValueError("batched field must have shape [batch, mode, channel]")
    if field.shape[1] < 2:
        return field
    left = np.concatenate([field[:, :1], field[:, :-1]], axis=1)
    right = np.concatenate([field[:, 1:], field[:, -1:]], axis=1)
    return field + rate * (left - 2.0 * field + right)


def _linear_field_operator(size: int, decay: float, rate: float) -> np.ndarray:
    """Return A for ``field_next = diffuse(decay * field, rate)``."""
    if size < 1:
        raise ValueError("field size must be positive")
    identity = np.eye(size, dtype=np.float64)
    return np.column_stack(
        [_diffuse((decay * identity[:, column])[:, None], rate)[:, 0]
        for column in range(size)]
    )


def _materialize_deferred_field(
    injections: np.ndarray,
    scales: np.ndarray,
    transition: np.ndarray,
) -> np.ndarray:
    """Materialize a linearly propagated field from its transaction-local deltas.

    The eager recurrence is ``z_t = scales[t] * (A z_(t-1) + u_t)``.
    It is equivalent, absent clipping, to the weighted finite sum evaluated
    here. The caller retains the scalar projections needed by feedback while
    deferring the full field until a successful commit.
    """
    if injections.ndim != 3:
        raise ValueError("injections must have shape [time, mode, channel]")
    steps, modes, channels = injections.shape
    if scales.shape != (steps,) or transition.shape != (modes, modes):
        raise ValueError("incompatible deferred-field shapes")
    powers = [np.eye(modes, dtype=np.float64)]
    for _ in range(1, steps):
        powers.append(powers[-1] @ transition)
    transforms = np.stack([powers[steps - 1 - step] for step in range(steps)])
    suffix_scales = np.cumprod(scales[::-1])[::-1]
    return np.einsum("t,tij,tjk->ik", suffix_scales, transforms, injections)


def _materialize_deferred_fields_batch(
    injections: np.ndarray,
    scales: np.ndarray,
    transition: np.ndarray,
) -> np.ndarray:
    """Batched counterpart of :func:`_materialize_deferred_field`."""
    if injections.ndim != 4:
        raise ValueError("injections must have shape [batch, time, mode, channel]")
    batch, steps, modes, _ = injections.shape
    if scales.shape != (batch, steps) or transition.shape != (modes, modes):
        raise ValueError("incompatible deferred-field batch shapes")
    powers = [np.eye(modes, dtype=np.float64)]
    for _ in range(1, steps):
        powers.append(powers[-1] @ transition)
    transforms = np.stack([powers[steps - 1 - step] for step in range(steps)])
    suffix_scales = np.cumprod(scales[:, ::-1], axis=1)[:, ::-1]
    return np.einsum("bt,tij,btjk->bik", suffix_scales, transforms, injections)


@dataclass(frozen=True)
class CommittedVirtualState:
    """Immutable state published only after a virtual sequence succeeds."""

    memory: np.ndarray
    relation: np.ndarray
    context: np.ndarray
    affect: np.ndarray
    self_state: np.ndarray
    semantic: np.ndarray


class NDL2FeedbackModel(Encoder):
    """Galerkin-discretized L2 fields with vector-valued feedback operators."""

    def _evolve_context(
        self,
        field: np.ndarray,
        x: np.ndarray,
        kind: np.ndarray,
        context: np.ndarray,
        step: int,
        seq_len: int,
    ) -> np.ndarray:
        """Ungated additive context evolution retained as the ablation."""
        del x, step, seq_len
        field = _diffuse(field * 0.945, 0.08)
        field[0] += context * max(0.0, kind[TYPE_CONTEXT])
        return field

    def encode_one(self, sequence: np.ndarray) -> np.ndarray:
        m = np.zeros((N_KEYS, N_VALUES))       # H_M
        r = np.zeros((N_KEYS, N_RELATIONS))    # H_R
        c = np.zeros((4, 4))                   # H_C
        e = np.zeros((8, 4))                   # H_E
        i = np.zeros((8, 4))                   # H_I
        s = np.zeros((5, 8))                   # H_S

        for step, x in enumerate(sequence):
            kind = np.clip(x[TYPE_SLICE], -0.25, 1.25)
            key = np.clip(x[KEY_SLICE], -0.25, 1.25)
            value = np.clip(x[VALUE_SLICE], -0.25, 1.25)
            rel = np.clip(x[RELATION_SLICE], -0.25, 1.25)
            ctx = np.clip(x[CONTEXT_SLICE], -0.25, 1.25)
            a = float(np.clip(x[AFFECT_INDEX], -1.5, 1.5))

            # Contractive field evolution plus task-independent local injections.
            m = _diffuse(m * 0.997, 0.012)
            r = _diffuse(r * 0.997, 0.012)
            c = self._evolve_context(c, x, kind, ctx, step, len(sequence))
            e = _diffuse(e * 0.982, 0.10)
            i = _diffuse(i * 0.998, 0.07)
            s = _diffuse(s * 0.972, 0.06)

            m += np.outer(key, value) * max(0.0, kind[TYPE_WRITE])
            r += np.outer(key, rel) * max(0.0, kind[TYPE_RELATION])
            basis_e = np.exp(-0.35 * np.arange(8))
            e[:, 0] += basis_e * a * max(0.0, kind[TYPE_AFFECT])
            e[:, 1] += basis_e * abs(a) * max(0.0, kind[TYPE_AFFECT])
            i[:, 0] += basis_e * a * max(0.0, kind[TYPE_AFFECT])
            i[:, 1] += basis_e * np.sign(a) * np.sqrt(abs(a)) * max(0.0, kind[TYPE_AFFECT])

            raw8 = np.array([
                np.mean(kind), np.mean(key), np.mean(value), np.mean(ctx),
                np.mean(rel), a, step / len(sequence), 1.0,
            ])
            s[0] += raw8 * 0.055

            # K_ij: field-to-field, vector-valued cross-feedback. Coefficients
            # are small so D + K remains contractive in this finite truncation.
            e[:, 2] += 0.018 * np.resize(c[0], 8)
            i[:, 2] += 0.022 * e[:, 0]
            i[:, 3] += 0.012 * e[:, 1]
            c[1] += 0.012 * np.resize(np.mean(s, axis=0), 4)
            s[1] += 0.010 * np.resize(np.mean(r, axis=0), 8)
            s[2] += 0.008 * np.mean(m, axis=0)
            gate = 1.0 + 0.003 * np.tanh(np.mean(i[:, 0]))
            m *= gate
            r *= gate

            for field in (m, r, c, e, i, s):
                np.clip(field, -4.0, 4.0, out=field)

        query_key = np.clip(sequence[-1, KEY_SLICE], 0.0, 1.25)
        retrieved_memory = query_key @ m
        retrieved_relation = query_key @ r
        # The six complete fields use 192 coefficients exactly; retrieval is
        # folded into their leading semantic coefficients without widening.
        s[-1, : retrieved_memory.size] += retrieved_memory
        s[-2, : retrieved_relation.size] += retrieved_relation
        return np.concatenate([m.ravel(), r.ravel(), c.ravel(), e.ravel(), i.ravel(), s.ravel()])


class NDL2GatedFeedbackModel(NDL2FeedbackModel):
    """ND-L2 model with a learned fast overwrite gate for context.

    The six gate coefficients are fitted on local context-write indicators in
    the training sequences. This is a small supervised gate, not end-to-end
    optimization of the full field dynamics.
    """

    trainable_internal_parameters = 6

    def __init__(self, *, learning_rate: float = 0.2, iterations: int = 600) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        self.learning_rate = float(learning_rate)
        self.iterations = int(iterations)
        self.gate_weights_: np.ndarray | None = None

    @staticmethod
    def _gate_features(sequence: np.ndarray) -> np.ndarray:
        kinds = sequence[:, TYPE_SLICE]
        contexts = sequence[:, CONTEXT_SLICE]
        return np.column_stack(
            [
                np.ones(len(sequence)),
                kinds[:, TYPE_CONTEXT],
                np.max(contexts, axis=1),
                np.linalg.norm(contexts, axis=1),
                np.sum(np.abs(contexts), axis=1),
                kinds[:, TYPE_QUERY],
            ]
        )

    def fit(
        self,
        sequences: np.ndarray,
        targets: dict[str, np.ndarray] | None = None,
    ) -> "NDL2GatedFeedbackModel":
        del targets
        if sequences.ndim != 3 or sequences.shape[-1] != INPUT_DIM:
            raise ValueError(f"expected [batch, time, {INPUT_DIM}] input")
        phi = np.vstack([self._gate_features(seq) for seq in sequences])
        raw_kinds = sequences[:, :, TYPE_SLICE].reshape(-1, TYPE_SLICE.stop)
        labels = (raw_kinds[:, TYPE_CONTEXT] > 0.5).astype(np.float64)
        positives = float(labels.sum())
        negatives = float(len(labels) - positives)
        if positives == 0 or negatives == 0:
            raise ValueError("training data must contain context and non-context tokens")
        sample_weights = np.where(
            labels > 0.5,
            len(labels) / (2.0 * positives),
            len(labels) / (2.0 * negatives),
        )
        weights = np.zeros(phi.shape[1], dtype=np.float64)
        for _ in range(self.iterations):
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(phi @ weights, -30.0, 30.0)))
            gradient = phi.T @ (sample_weights * (probabilities - labels)) / len(labels)
            gradient[1:] += 1e-4 * weights[1:]
            weights -= self.learning_rate * gradient
        self.gate_weights_ = weights
        return self

    def gate_probabilities(self, sequence: np.ndarray) -> np.ndarray:
        if self.gate_weights_ is None:
            raise RuntimeError("context gate must be fitted before encoding")
        logits = self._gate_features(sequence) @ self.gate_weights_
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def _evolve_context(
        self,
        field: np.ndarray,
        x: np.ndarray,
        kind: np.ndarray,
        context: np.ndarray,
        step: int,
        seq_len: int,
    ) -> np.ndarray:
        del kind, seq_len
        if self.gate_weights_ is None:
            raise RuntimeError("context gate must be fitted before encoding")
        field = _diffuse(field * 0.997, 0.02)
        del step
        logits = self._gate_features(x[None, :]) @ self.gate_weights_
        gate = float(1.0 / (1.0 + np.exp(-np.clip(logits[0], -30.0, 30.0))))
        basis = np.array([1.0, 0.50, 0.25, 0.125])
        candidate = np.outer(basis, context)
        return (1.0 - gate) * field + gate * candidate


class NDL2VirtualGatedFeedbackModel(NDL2GatedFeedbackModel):
    """Gated ND-L2 with transaction-local deltas and lazy field materialization.

    Memory and relation fields are represented virtually while their exact mean
    projections remain available to the nonlinear feedback loop. Full fields
    are materialized and published only after the sequence completed without
    error. This is algebraically equivalent to the eager model while clipping
    is inactive; the implementation verifies that safety envelope at commit.
    """

    def __init__(self, **kwargs: float | int) -> None:
        super().__init__(**kwargs)
        self.last_committed_state_: CommittedVirtualState | None = None
        self._memory_transition = _linear_field_operator(N_KEYS, 0.997, 0.012)
        self._relation_transition = _linear_field_operator(N_KEYS, 0.997, 0.012)

    @staticmethod
    def _require_sequence(sequence: np.ndarray) -> None:
        if sequence.ndim != 2 or sequence.shape[1] != INPUT_DIM:
            raise ValueError(f"expected [time, {INPUT_DIM}] input")

    def encode_one(self, sequence: np.ndarray) -> np.ndarray:
        self._require_sequence(sequence)
        if self.gate_weights_ is None:
            raise RuntimeError("context gate must be fitted before encoding")

        steps = len(sequence)
        kinds = np.clip(sequence[:, TYPE_SLICE], -0.25, 1.25)
        keys = np.clip(sequence[:, KEY_SLICE], -0.25, 1.25)
        values = np.clip(sequence[:, VALUE_SLICE], -0.25, 1.25)
        relations = np.clip(sequence[:, RELATION_SLICE], -0.25, 1.25)
        contexts = np.clip(sequence[:, CONTEXT_SLICE], -0.25, 1.25)
        affects = np.clip(sequence[:, AFFECT_INDEX], -1.5, 1.5)
        gates = self.gate_probabilities(sequence)

        memory_injections = np.einsum(
            "ti,tj,t->tij", keys, values, np.maximum(kinds[:, TYPE_WRITE], 0.0)
        )
        relation_injections = np.einsum(
            "ti,tj,t->tij", keys, relations, np.maximum(kinds[:, TYPE_RELATION], 0.0)
        )
        memory_mean = np.zeros(N_VALUES)
        relation_mean = np.zeros(N_RELATIONS)
        scales = np.empty(steps)

        c = np.zeros((4, 4))
        e = np.zeros((8, 4))
        i = np.zeros((8, 4))
        s = np.zeros((5, 8))
        basis_e = np.exp(-0.35 * np.arange(8))

        for step in range(steps):
            kind = kinds[step]
            key = keys[step]
            value = values[step]
            rel = relations[step]
            ctx = contexts[step]
            a = float(affects[step])

            # The fast context field is still immediately visible to feedback.
            c = _diffuse(c * 0.997, 0.02)
            candidate = np.outer(np.array([1.0, 0.50, 0.25, 0.125]), ctx)
            c = (1.0 - gates[step]) * c + gates[step] * candidate
            e = _diffuse(e * 0.982, 0.10)
            i = _diffuse(i * 0.998, 0.07)
            s = _diffuse(s * 0.972, 0.06)

            e[:, 0] += basis_e * a * max(0.0, kind[TYPE_AFFECT])
            e[:, 1] += basis_e * abs(a) * max(0.0, kind[TYPE_AFFECT])
            i[:, 0] += basis_e * a * max(0.0, kind[TYPE_AFFECT])
            i[:, 1] += basis_e * np.sign(a) * np.sqrt(abs(a)) * max(0.0, kind[TYPE_AFFECT])

            # Diffusion has zero net flux, hence the field mean follows this
            # scalar recurrence exactly and can drive feedback without the full
            # memory/relation tensors being materialized.
            memory_pre = 0.997 * memory_mean + memory_injections[step].mean(axis=0)
            relation_pre = 0.997 * relation_mean + relation_injections[step].mean(axis=0)

            raw8 = np.array([
                np.mean(kind), np.mean(key), np.mean(value), np.mean(ctx),
                np.mean(rel), a, step / steps, 1.0,
            ])
            s[0] += raw8 * 0.055
            e[:, 2] += 0.018 * np.resize(c[0], 8)
            i[:, 2] += 0.022 * e[:, 0]
            i[:, 3] += 0.012 * e[:, 1]
            c[1] += 0.012 * np.resize(np.mean(s, axis=0), 4)
            s[1] += 0.010 * np.resize(relation_pre, 8)
            s[2] += 0.008 * memory_pre
            scale = 1.0 + 0.003 * np.tanh(np.mean(i[:, 0]))
            scales[step] = scale
            memory_mean = scale * memory_pre
            relation_mean = scale * relation_pre

            for field in (c, e, i, s):
                np.clip(field, -4.0, 4.0, out=field)

        memory = _materialize_deferred_field(
            memory_injections, scales, self._memory_transition
        )
        relation = _materialize_deferred_field(
            relation_injections, scales, self._relation_transition
        )
        if max(np.max(np.abs(memory)), np.max(np.abs(relation))) > 4.0:
            raise RuntimeError("virtual field escaped the eager model clipping envelope")

        query_key = np.clip(sequence[-1, KEY_SLICE], 0.0, 1.25)
        semantic = s.copy()
        semantic[-1, :N_VALUES] += query_key @ memory
        semantic[-2, :N_RELATIONS] += query_key @ relation
        committed = CommittedVirtualState(
            memory=memory.copy(), relation=relation.copy(), context=c.copy(),
            affect=e.copy(), self_state=i.copy(), semantic=semantic.copy(),
        )
        # Publish only after all validation/materialization steps completed.
        self.last_committed_state_ = committed
        return np.concatenate(
            [memory.ravel(), relation.ravel(), c.ravel(), e.ravel(), i.ravel(), semantic.ravel()]
        )

    def encode(self, sequences: np.ndarray) -> np.ndarray:
        """Vectorized, numerically equivalent batch execution.

        Recurrence is retained along time, while every field update executes
        across the whole batch. This removes the per-sequence Python loop but
        leaves the state equation unchanged.
        """
        if sequences.ndim != 3 or sequences.shape[-1] != INPUT_DIM:
            raise ValueError(f"expected [batch, time, {INPUT_DIM}] input")
        if self.gate_weights_ is None:
            raise RuntimeError("context gate must be fitted before encoding")

        batch, steps, _ = sequences.shape
        kinds = np.clip(sequences[:, :, TYPE_SLICE], -0.25, 1.25)
        keys = np.clip(sequences[:, :, KEY_SLICE], -0.25, 1.25)
        values = np.clip(sequences[:, :, VALUE_SLICE], -0.25, 1.25)
        relations = np.clip(sequences[:, :, RELATION_SLICE], -0.25, 1.25)
        contexts = np.clip(sequences[:, :, CONTEXT_SLICE], -0.25, 1.25)
        affects = np.clip(sequences[:, :, AFFECT_INDEX], -1.5, 1.5)

        raw_contexts = sequences[:, :, CONTEXT_SLICE]
        gate_features = np.stack(
            [
                np.ones((batch, steps)),
                sequences[:, :, TYPE_CONTEXT],
                np.max(raw_contexts, axis=2),
                np.linalg.norm(raw_contexts, axis=2),
                np.sum(np.abs(raw_contexts), axis=2),
                sequences[:, :, TYPE_QUERY],
            ],
            axis=2,
        )
        gate_logits = np.einsum("btd,d->bt", gate_features, self.gate_weights_)
        gates = 1.0 / (1.0 + np.exp(-np.clip(gate_logits, -30.0, 30.0)))

        memory_injections = np.einsum(
            "bti,btj,bt->btij", keys, values, np.maximum(kinds[:, :, TYPE_WRITE], 0.0)
        )
        relation_injections = np.einsum(
            "bti,btj,bt->btij", keys, relations, np.maximum(kinds[:, :, TYPE_RELATION], 0.0)
        )
        memory_mean = np.zeros((batch, N_VALUES))
        relation_mean = np.zeros((batch, N_RELATIONS))
        scales = np.empty((batch, steps))

        c = np.zeros((batch, 4, 4))
        e = np.zeros((batch, 8, 4))
        i = np.zeros((batch, 8, 4))
        s = np.zeros((batch, 5, 8))
        basis_e = np.exp(-0.35 * np.arange(8))[None, :]
        context_basis = np.array([1.0, 0.50, 0.25, 0.125])[None, :, None]

        for step in range(steps):
            kind = kinds[:, step]
            key = keys[:, step]
            value = values[:, step]
            rel = relations[:, step]
            ctx = contexts[:, step]
            a = affects[:, step]

            c = _diffuse_batch(c * 0.997, 0.02)
            candidate = context_basis * ctx[:, None, :]
            gate = gates[:, step, None, None]
            c = (1.0 - gate) * c + gate * candidate
            e = _diffuse_batch(e * 0.982, 0.10)
            i = _diffuse_batch(i * 0.998, 0.07)
            s = _diffuse_batch(s * 0.972, 0.06)

            affect_write = np.maximum(kind[:, TYPE_AFFECT], 0.0)[:, None]
            e[:, :, 0] += basis_e * a[:, None] * affect_write
            e[:, :, 1] += basis_e * np.abs(a)[:, None] * affect_write
            i[:, :, 0] += basis_e * a[:, None] * affect_write
            i[:, :, 1] += basis_e * np.sign(a)[:, None] * np.sqrt(np.abs(a))[:, None] * affect_write

            memory_pre = 0.997 * memory_mean + memory_injections[:, step].mean(axis=1)
            relation_pre = 0.997 * relation_mean + relation_injections[:, step].mean(axis=1)
            raw8 = np.column_stack(
                [
                    np.mean(kind, axis=1), np.mean(key, axis=1), np.mean(value, axis=1),
                    np.mean(ctx, axis=1), np.mean(rel, axis=1), a,
                    np.full(batch, step / steps), np.ones(batch),
                ]
            )
            s[:, 0] += raw8 * 0.055
            e[:, :, 2] += 0.018 * np.tile(c[:, 0], (1, 2))
            i[:, :, 2] += 0.022 * e[:, :, 0]
            i[:, :, 3] += 0.012 * e[:, :, 1]
            c[:, 1] += 0.012 * np.mean(s, axis=1)[:, :4]
            s[:, 1] += 0.010 * np.tile(relation_pre, (1, 2))
            s[:, 2] += 0.008 * memory_pre
            scale = 1.0 + 0.003 * np.tanh(np.mean(i[:, :, 0], axis=1))
            scales[:, step] = scale
            memory_mean = scale[:, None] * memory_pre
            relation_mean = scale[:, None] * relation_pre
            for field in (c, e, i, s):
                np.clip(field, -4.0, 4.0, out=field)

        memory = _materialize_deferred_fields_batch(
            memory_injections, scales, self._memory_transition
        )
        relation = _materialize_deferred_fields_batch(
            relation_injections, scales, self._relation_transition
        )
        if max(np.max(np.abs(memory)), np.max(np.abs(relation))) > 4.0:
            raise RuntimeError("virtual field escaped the eager model clipping envelope")

        query_keys = np.clip(sequences[:, -1, KEY_SLICE], 0.0, 1.25)
        semantic = s.copy()
        semantic[:, -1, :N_VALUES] += np.einsum("bi,bij->bj", query_keys, memory)
        semantic[:, -2, :N_RELATIONS] += np.einsum("bi,bij->bj", query_keys, relation)
        committed_states = [
            CommittedVirtualState(
                memory=memory[index].copy(), relation=relation[index].copy(),
                context=c[index].copy(), affect=e[index].copy(),
                self_state=i[index].copy(), semantic=semantic[index].copy(),
            )
            for index in range(batch)
        ]
        self.last_committed_states_ = tuple(committed_states)
        self.last_committed_state_ = committed_states[-1] if committed_states else None
        return np.concatenate(
            [
                memory.reshape(batch, -1), relation.reshape(batch, -1), c.reshape(batch, -1),
                e.reshape(batch, -1), i.reshape(batch, -1), semantic.reshape(batch, -1),
            ],
            axis=1,
        )


class NDL2JitVirtualGatedFeedbackModel(NDL2VirtualGatedFeedbackModel):
    """Virtual-state model whose remaining time recurrence uses Numba JIT."""

    def __init__(self, **kwargs: float | int) -> None:
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba is unavailable; install the 'jit' optional dependency")
        super().__init__(**kwargs)

    def encode(self, sequences: np.ndarray) -> np.ndarray:
        if sequences.ndim != 3 or sequences.shape[-1] != INPUT_DIM:
            raise ValueError(f"expected [batch, time, {INPUT_DIM}] input")
        if self.gate_weights_ is None:
            raise RuntimeError("context gate must be fitted before encoding")
        batch, steps, _ = sequences.shape
        kinds = np.clip(sequences[:, :, TYPE_SLICE], -0.25, 1.25)
        keys = np.clip(sequences[:, :, KEY_SLICE], -0.25, 1.25)
        values = np.clip(sequences[:, :, VALUE_SLICE], -0.25, 1.25)
        relations = np.clip(sequences[:, :, RELATION_SLICE], -0.25, 1.25)
        contexts = np.clip(sequences[:, :, CONTEXT_SLICE], -0.25, 1.25)
        affects = np.clip(sequences[:, :, AFFECT_INDEX], -1.5, 1.5)
        raw_contexts = sequences[:, :, CONTEXT_SLICE]
        gate_features = np.stack(
            [
                np.ones((batch, steps)), sequences[:, :, TYPE_CONTEXT],
                np.max(raw_contexts, axis=2), np.linalg.norm(raw_contexts, axis=2),
                np.sum(np.abs(raw_contexts), axis=2), sequences[:, :, TYPE_QUERY],
            ],
            axis=2,
        )
        gate_logits = np.einsum("btd,d->bt", gate_features, self.gate_weights_)
        gates = 1.0 / (1.0 + np.exp(-np.clip(gate_logits, -30.0, 30.0)))
        memory_injections = np.einsum(
            "bti,btj,bt->btij", keys, values, np.maximum(kinds[:, :, TYPE_WRITE], 0.0)
        )
        relation_injections = np.einsum(
            "bti,btj,bt->btij", keys, relations, np.maximum(kinds[:, :, TYPE_RELATION], 0.0)
        )
        c, e, i, s, scales = run_virtual_recurrence(
            kinds, keys, values, relations, contexts, affects, gates,
            memory_injections, relation_injections,
        )
        memory = _materialize_deferred_fields_batch(
            memory_injections, scales, self._memory_transition
        )
        relation = _materialize_deferred_fields_batch(
            relation_injections, scales, self._relation_transition
        )
        if max(np.max(np.abs(memory)), np.max(np.abs(relation))) > 4.0:
            raise RuntimeError("virtual field escaped the eager model clipping envelope")
        query_keys = np.clip(sequences[:, -1, KEY_SLICE], 0.0, 1.25)
        semantic = s.copy()
        semantic[:, -1, :N_VALUES] += np.einsum("bi,bij->bj", query_keys, memory)
        semantic[:, -2, :N_RELATIONS] += np.einsum("bi,bij->bj", query_keys, relation)
        committed_states = [
            CommittedVirtualState(
                memory=memory[index].copy(), relation=relation[index].copy(),
                context=c[index].copy(), affect=e[index].copy(),
                self_state=i[index].copy(), semantic=semantic[index].copy(),
            )
            for index in range(batch)
        ]
        self.last_committed_states_ = tuple(committed_states)
        self.last_committed_state_ = committed_states[-1] if committed_states else None
        return np.concatenate(
            [
                memory.reshape(batch, -1), relation.reshape(batch, -1), c.reshape(batch, -1),
                e.reshape(batch, -1), i.reshape(batch, -1), semantic.reshape(batch, -1),
            ],
            axis=1,
        )
