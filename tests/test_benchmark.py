import unittest

import numpy as np

from nd_l2_benchmark.data import INPUT_DIM, generate_dataset, perturb
from nd_l2_benchmark.metrics import RidgeReadout, accuracies
from nd_l2_benchmark.models import (
    CausalAttentionBaseline,
    FEATURE_DIM,
    NDL2FeedbackModel,
    NDL2GatedFeedbackModel,
    NDL2VirtualGatedFeedbackModel,
    NDL2JitVirtualGatedFeedbackModel,
    NDStateModel,
)
from nd_l2_benchmark.jit_backend import NUMBA_AVAILABLE


class BenchmarkTests(unittest.TestCase):
    def test_dataset_is_deterministic_and_shaped(self):
        a = generate_dataset(12, seed=7)
        b = generate_dataset(12, seed=7)
        self.assertEqual(a.x.shape, (12, 36, INPUT_DIM))
        np.testing.assert_allclose(a.x, b.x)
        np.testing.assert_array_equal(a.memory, b.memory)

    def test_all_models_have_equal_finite_width(self):
        data = generate_dataset(4, seed=3)
        for model in (CausalAttentionBaseline(), NDStateModel(), NDL2FeedbackModel()):
            encoded = model.encode(data.x)
            self.assertEqual(encoded.shape, (4, FEATURE_DIM))
            self.assertTrue(np.isfinite(encoded).all())

    def test_noise_keeps_labels(self):
        data = generate_dataset(8, seed=2)
        noisy = perturb(data, sigma=0.1, seed=9)
        self.assertFalse(np.array_equal(data.x, noisy.x))
        for name, target in data.targets.items():
            np.testing.assert_array_equal(target, noisy.targets[name])

    def test_context_label_matches_latest_context_event(self):
        data = generate_dataset(30, seed=21)
        from nd_l2_benchmark.data import CONTEXT_SLICE, TYPE_CONTEXT
        for sequence, expected in zip(data.x, data.context):
            positions = np.where(sequence[:, TYPE_CONTEXT] > 0.5)[0]
            self.assertEqual(len(positions), 3)
            latest = sequence[positions[-1], CONTEXT_SLICE]
            self.assertEqual(int(np.argmax(latest)), int(expected))

    def test_readout_learns_nontrivial_signal(self):
        train = generate_dataset(180, seed=11)
        test = generate_dataset(80, seed=12)
        model = NDL2FeedbackModel()
        readout = RidgeReadout(alpha=2.0).fit(model.encode(train.x), train.targets)
        scores = accuracies(readout.predict(model.encode(test.x)), test.targets)
        self.assertGreater(scores["macro"], 0.40)

    def test_learned_gate_separates_context_events(self):
        train = generate_dataset(80, seed=31)
        model = NDL2GatedFeedbackModel(iterations=300).fit(train.x)
        probs = np.concatenate([model.gate_probabilities(seq) for seq in train.x])
        kinds = train.x[:, :, 1].reshape(-1)
        self.assertGreater(float(probs[kinds > 0.5].mean()), 0.90)
        self.assertLess(float(probs[kinds < 0.5].mean()), 0.10)

    def test_gated_model_requires_fit(self):
        data = generate_dataset(2, seed=33)
        with self.assertRaises(RuntimeError):
            NDL2GatedFeedbackModel().encode(data.x)

    def test_gated_model_has_equal_feature_width(self):
        data = generate_dataset(12, seed=35)
        model = NDL2GatedFeedbackModel(iterations=300).fit(data.x)
        encoded = model.encode(data.x)
        self.assertEqual(encoded.shape, (12, FEATURE_DIM))
        self.assertTrue(np.isfinite(encoded).all())

    def test_virtual_state_matches_eager_gated_model_and_commits(self):
        train = generate_dataset(90, seed=41)
        probe = generate_dataset(12, seed=42)
        eager = NDL2GatedFeedbackModel(iterations=300).fit(train.x)
        virtual = NDL2VirtualGatedFeedbackModel(iterations=300).fit(train.x)
        np.testing.assert_allclose(
            virtual.encode(probe.x), eager.encode(probe.x), rtol=1e-10, atol=1e-10
        )
        self.assertIsNotNone(virtual.last_committed_state_)
        self.assertEqual(virtual.last_committed_state_.memory.shape, (6, 8))

    def test_virtual_state_keeps_previous_commit_on_failed_input(self):
        train = generate_dataset(30, seed=51)
        model = NDL2VirtualGatedFeedbackModel(iterations=300).fit(train.x)
        model.encode(train.x[:1])
        previous = model.last_committed_state_
        with self.assertRaises(ValueError):
            model.encode_one(np.zeros((5, INPUT_DIM - 1)))
        self.assertIs(model.last_committed_state_, previous)

    def test_vectorized_virtual_state_matches_single_sequence_execution(self):
        train = generate_dataset(70, seed=61)
        probe = generate_dataset(11, seed=62)
        model = NDL2VirtualGatedFeedbackModel(iterations=300).fit(train.x)
        batched = model.encode(probe.x)
        individual = np.vstack([model.encode_one(sequence) for sequence in probe.x])
        np.testing.assert_allclose(batched, individual, rtol=1e-10, atol=1e-10)
        self.assertEqual(len(model.last_committed_states_), len(probe.x))

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba JIT optional dependency is unavailable")
    def test_jit_virtual_state_matches_vectorized_state(self):
        train = generate_dataset(60, seed=71)
        probe = generate_dataset(9, seed=72)
        vectorized = NDL2VirtualGatedFeedbackModel(iterations=300).fit(train.x)
        jit = NDL2JitVirtualGatedFeedbackModel(iterations=300).fit(train.x)
        np.testing.assert_allclose(
            jit.encode(probe.x), vectorized.encode(probe.x), rtol=1e-10, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
