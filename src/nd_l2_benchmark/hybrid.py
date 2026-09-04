"""GitHub-Action-ready hybrid LLM and transactional state prototype.

One large LLM call returns both a user-facing answer and a structured semantic
event. The local state engine validates and transactionally commits that event.
A future lightweight learned event encoder can implement the same provider
protocol without changing the durable-state boundary.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


VALID_OPERATIONS = {"assertion", "correction", "retraction", "none"}


@dataclass(frozen=True)
class SemanticEvent:
    operation: str = "none"
    subject: str = ""
    value: str = ""
    old_value: str = ""
    confidence: float = 0.0
    rationale: str = ""

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SemanticEvent":
        operation = str(payload.get("operation", "none")).strip().lower()
        if operation not in VALID_OPERATIONS:
            raise ValueError(f"unsupported semantic operation: {operation!r}")
        confidence = float(payload.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("event confidence must be between 0 and 1")
        return cls(
            operation=operation,
            subject=str(payload.get("subject", "")).strip(),
            value=str(payload.get("value", "")).strip(),
            old_value=str(payload.get("old_value", "")).strip(),
            confidence=confidence,
            rationale=str(payload.get("rationale", "")).strip(),
        )


@dataclass(frozen=True)
class LLMResult:
    answer: str
    event: SemanticEvent


class EventAnswerProvider(Protocol):
    def answer_and_extract(self, message: str, state: "HybridState") -> LLMResult: ...


@dataclass
class HybridState:
    version: int = 0
    facts: dict[str, dict[str, Any]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "HybridState":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("state file must contain a JSON object")
        facts = payload.get("facts", {})
        history = payload.get("history", [])
        if not isinstance(facts, dict) or not isinstance(history, list):
            raise ValueError("state file has invalid facts or history")
        return cls(version=int(payload.get("version", 0)), facts=facts, history=history)

    def snapshot(self) -> dict[str, Any]:
        return {"version": self.version, "facts": self.facts, "history": self.history[-12:]}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


@dataclass(frozen=True)
class StateDecision:
    accepted: bool
    gate: float
    reason: str
    next_state: HybridState


class HybridStateEngine:
    """Validate a semantic proposal before the active state may change."""

    def propose(self, state: HybridState, event: SemanticEvent) -> StateDecision:
        next_state = HybridState(
            version=state.version,
            facts={key: dict(value) for key, value in state.facts.items()},
            history=list(state.history),
        )
        if event.operation == "none" or not event.subject:
            return StateDecision(False, 0.0, "no state-changing semantic event", next_state)

        existing = next_state.facts.get(event.subject)
        conflict = bool(existing and existing.get("value") != event.value)
        gate = event.confidence
        if event.operation == "assertion" and conflict:
            gate *= 0.45
        if event.operation == "retraction":
            gate *= 0.90
        accepted = gate >= 0.50
        record = {
            "event": asdict(event), "gate": gate, "accepted": accepted,
            "previous": existing,
        }
        next_state.history.append(record)
        if not accepted:
            return StateDecision(False, gate, "gate below commit threshold", next_state)

        if event.operation == "retraction":
            next_state.facts.pop(event.subject, None)
        else:
            if event.operation == "correction" and event.old_value and existing:
                if existing.get("value") != event.old_value:
                    record["warning"] = "correction old_value differs from active fact"
            next_state.facts[event.subject] = {
                "value": event.value,
                "confidence": event.confidence,
                "operation": event.operation,
                "rationale": event.rationale,
            }
        next_state.version += 1
        return StateDecision(True, gate, "committed virtual update", next_state)


class OpenAIEventAnswerProvider:
    """One OpenAI Responses API call for answer plus semantic extraction."""

    def __init__(self, model: str) -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        self.model = model

    def answer_and_extract(self, message: str, state: HybridState) -> LLMResult:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised in GitHub Actions
            raise RuntimeError("install the 'openai' optional dependency") from exc
        instruction = (
            "Return ONLY one JSON object with keys answer and event. event must have "
            "operation (assertion, correction, retraction, none), subject, value, old_value, "
            "confidence (0..1), rationale. Use correction only when the user supersedes "
            "an active fact. Do not invent facts. State snapshot:\n"
            + json.dumps(state.snapshot(), ensure_ascii=False)
        )
        response = OpenAI().responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": message},
            ],
        )
        try:
            payload = json.loads(response.output_text)
            answer = str(payload["answer"]).strip()
            if not answer:
                raise ValueError("empty answer")
            event_payload = payload.get("event", {})
            if not isinstance(event_payload, dict):
                raise ValueError("event must be an object")
            return LLMResult(answer=answer, event=SemanticEvent.from_mapping(event_payload))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("LLM response was not valid hybrid JSON") from exc


class ScriptedProvider:
    """Offline provider for deterministic tests without API access."""

    def __init__(self, answer: str, event: SemanticEvent) -> None:
        self.result = LLMResult(answer=answer, event=event)

    def answer_and_extract(self, message: str, state: HybridState) -> LLMResult:
        del message, state
        return self.result


def process_turn(
    message: str,
    state: HybridState,
    provider: EventAnswerProvider,
    engine: HybridStateEngine | None = None,
) -> tuple[LLMResult, StateDecision]:
    if not message.strip():
        raise ValueError("message must be non-empty")
    result = provider.answer_and_extract(message, state)
    decision = (engine or HybridStateEngine()).propose(state, result.event)
    return result, decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", required=True)
    parser.add_argument("--state-file", type=Path, default=Path(".hybrid/state.json"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be configured by the runtime")
    state = HybridState.load(args.state_file)
    result, decision = process_turn(args.message, state, OpenAIEventAnswerProvider(args.model))
    decision.next_state.save(args.state_file)
    print(json.dumps({
        "answer": result.answer,
        "event": asdict(result.event),
        "accepted": decision.accepted,
        "gate": decision.gate,
        "reason": decision.reason,
        "state_version": decision.next_state.version,
        "state_file": str(args.state_file),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
