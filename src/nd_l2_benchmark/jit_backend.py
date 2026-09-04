"""Optional Numba kernel for the vectorized virtual-state recurrence."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without optional extra
    NUMBA_AVAILABLE = False

    def run_virtual_recurrence(*_args: np.ndarray) -> tuple[np.ndarray, ...]:
        raise RuntimeError("Numba is unavailable; install the 'jit' optional dependency")

else:
    NUMBA_AVAILABLE = True

    @njit(cache=True)
    def _diffuse(source: np.ndarray, target: np.ndarray, decay: float, rate: float) -> None:
        for batch in range(source.shape[0]):
            for mode in range(source.shape[1]):
                left_mode = mode - 1 if mode > 0 else 0
                right_mode = mode + 1 if mode + 1 < source.shape[1] else source.shape[1] - 1
                for channel in range(source.shape[2]):
                    center = decay * source[batch, mode, channel]
                    left = decay * source[batch, left_mode, channel]
                    right = decay * source[batch, right_mode, channel]
                    target[batch, mode, channel] = center + rate * (left - 2.0 * center + right)

    @njit(cache=True)
    def _clip_inplace(field: np.ndarray) -> None:
        for batch in range(field.shape[0]):
            for mode in range(field.shape[1]):
                for channel in range(field.shape[2]):
                    value = field[batch, mode, channel]
                    if value < -4.0:
                        field[batch, mode, channel] = -4.0
                    elif value > 4.0:
                        field[batch, mode, channel] = 4.0

    @njit(cache=True)
    def run_virtual_recurrence(
        kinds: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        relations: np.ndarray,
        contexts: np.ndarray,
        affects: np.ndarray,
        gates: np.ndarray,
        memory_injections: np.ndarray,
        relation_injections: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Native equivalent of the vectorized virtual state time recurrence."""
        batch_size, steps, _ = kinds.shape
        c = np.zeros((batch_size, 4, 4))
        e = np.zeros((batch_size, 8, 4))
        i = np.zeros((batch_size, 8, 4))
        s = np.zeros((batch_size, 5, 8))
        next_c = np.empty_like(c)
        next_e = np.empty_like(e)
        next_i = np.empty_like(i)
        next_s = np.empty_like(s)
        memory_mean = np.zeros((batch_size, 8))
        relation_mean = np.zeros((batch_size, 4))
        scales = np.empty((batch_size, steps))
        basis_e = np.empty(8)
        for mode in range(8):
            basis_e[mode] = np.exp(-0.35 * mode)
        context_basis = np.array((1.0, 0.50, 0.25, 0.125))

        for step in range(steps):
            _diffuse(c, next_c, 0.997, 0.02)
            _diffuse(e, next_e, 0.982, 0.10)
            _diffuse(i, next_i, 0.998, 0.07)
            _diffuse(s, next_s, 0.972, 0.06)
            c, next_c = next_c, c
            e, next_e = next_e, e
            i, next_i = next_i, i
            s, next_s = next_s, s

            for batch in range(batch_size):
                gate = gates[batch, step]
                for mode in range(4):
                    for channel in range(4):
                        candidate = context_basis[mode] * contexts[batch, step, channel]
                        c[batch, mode, channel] = (1.0 - gate) * c[batch, mode, channel] + gate * candidate

                affect_write = kinds[batch, step, 2]
                if affect_write < 0.0:
                    affect_write = 0.0
                affect = affects[batch, step]
                for mode in range(8):
                    e[batch, mode, 0] += basis_e[mode] * affect * affect_write
                    e[batch, mode, 1] += basis_e[mode] * abs(affect) * affect_write
                    i[batch, mode, 0] += basis_e[mode] * affect * affect_write
                    i[batch, mode, 1] += basis_e[mode] * np.sign(affect) * np.sqrt(abs(affect)) * affect_write

                kind_mean = 0.0
                key_mean = 0.0
                value_mean = 0.0
                context_mean = 0.0
                relation_mean_raw = 0.0
                for channel in range(6):
                    kind_mean += kinds[batch, step, channel]
                    key_mean += keys[batch, step, channel]
                for channel in range(8):
                    value_mean += values[batch, step, channel]
                for channel in range(4):
                    context_mean += contexts[batch, step, channel]
                    relation_mean_raw += relations[batch, step, channel]
                raw = np.array((
                    kind_mean / 6.0, key_mean / 6.0, value_mean / 8.0,
                    context_mean / 4.0, relation_mean_raw / 4.0, affect,
                    step / steps, 1.0,
                ))
                for channel in range(8):
                    s[batch, 0, channel] += raw[channel] * 0.055

                for channel in range(8):
                    e[batch, channel, 2] += 0.018 * c[batch, 0, channel % 4]
                    i[batch, channel, 2] += 0.022 * e[batch, channel, 0]
                    i[batch, channel, 3] += 0.012 * e[batch, channel, 1]

                for channel in range(8):
                    injection_mean = 0.0
                    for mode in range(6):
                        injection_mean += memory_injections[batch, step, mode, channel]
                    memory_mean[batch, channel] = 0.997 * memory_mean[batch, channel] + injection_mean / 6.0
                for channel in range(4):
                    injection_mean = 0.0
                    for mode in range(6):
                        injection_mean += relation_injections[batch, step, mode, channel]
                    relation_mean[batch, channel] = 0.997 * relation_mean[batch, channel] + injection_mean / 6.0

                for channel in range(4):
                    semantic_mean = 0.0
                    for mode in range(5):
                        semantic_mean += s[batch, mode, channel]
                    c[batch, 1, channel] += 0.012 * semantic_mean / 5.0
                for channel in range(8):
                    s[batch, 1, channel] += 0.010 * relation_mean[batch, channel % 4]
                    s[batch, 2, channel] += 0.008 * memory_mean[batch, channel]

                self_mean = 0.0
                for mode in range(8):
                    self_mean += i[batch, mode, 0]
                scale = 1.0 + 0.003 * np.tanh(self_mean / 8.0)
                scales[batch, step] = scale
                for channel in range(8):
                    memory_mean[batch, channel] *= scale
                for channel in range(4):
                    relation_mean[batch, channel] *= scale

            _clip_inplace(c)
            _clip_inplace(e)
            _clip_inplace(i)
            _clip_inplace(s)
        return c, e, i, s, scales
