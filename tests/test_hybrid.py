import tempfile
import unittest
from pathlib import Path

from nd_l2_benchmark.hybrid import (
    HybridState,
    HybridStateEngine,
    ScriptedProvider,
    SemanticEvent,
    process_turn,
)


class HybridTests(unittest.TestCase):
    def test_correction_commits_current_fact_and_preserves_history(self):
        state = HybridState(
            version=1,
            facts={"Termin": {"value": "Donnerstag", "confidence": 0.9}},
        )
        provider = ScriptedProvider(
            "Der Termin ist Freitag.",
            SemanticEvent("correction", "Termin", "Freitag", "Donnerstag", 0.96, "verschoben"),
        )
        result, decision = process_turn("Er wurde auf Freitag verschoben.", state, provider)
        self.assertTrue(decision.accepted)
        self.assertEqual(result.answer, "Der Termin ist Freitag.")
        self.assertEqual(decision.next_state.facts["Termin"]["value"], "Freitag")
        self.assertEqual(len(decision.next_state.history), 1)
        self.assertEqual(state.facts["Termin"]["value"], "Donnerstag")

    def test_low_confidence_conflicting_assertion_does_not_overwrite(self):
        state = HybridState(facts={"Termin": {"value": "Freitag", "confidence": 0.9}})
        decision = HybridStateEngine().propose(
            state, SemanticEvent("assertion", "Termin", "Montag", "", 0.80, "unbestätigt")
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.next_state.facts["Termin"]["value"], "Freitag")

    def test_state_save_and_load_are_atomic_at_the_api_boundary(self):
        state = HybridState(version=2, facts={"Ort": {"value": "Berlin"}})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            state.save(path)
            restored = HybridState.load(path)
        self.assertEqual(restored.version, 2)
        self.assertEqual(restored.facts["Ort"]["value"], "Berlin")


if __name__ == "__main__":
    unittest.main()
