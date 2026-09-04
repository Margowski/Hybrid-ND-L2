"""ND-L2 cognitive state benchmark."""

from typing import TYPE_CHECKING

from .data import BenchmarkData, generate_dataset
from .models import (
    CausalAttentionBaseline,
    NDStateModel,
    NDL2FeedbackModel,
    NDL2GatedFeedbackModel,
    NDL2VirtualGatedFeedbackModel,
    NDL2JitVirtualGatedFeedbackModel,
)

if TYPE_CHECKING:
    from .hybrid import HybridState, HybridStateEngine, SemanticEvent


def __getattr__(name: str):
    """Load optional hybrid types lazily to keep its CLI runnable as a module."""
    if name in {"HybridState", "HybridStateEngine", "SemanticEvent"}:
        from .hybrid import HybridState, HybridStateEngine, SemanticEvent

        return {
            "HybridState": HybridState,
            "HybridStateEngine": HybridStateEngine,
            "SemanticEvent": SemanticEvent,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BenchmarkData",
    "generate_dataset",
    "CausalAttentionBaseline",
    "NDStateModel",
    "NDL2FeedbackModel",
    "NDL2GatedFeedbackModel",
    "NDL2VirtualGatedFeedbackModel",
    "NDL2JitVirtualGatedFeedbackModel",
    "HybridState",
    "HybridStateEngine",
    "SemanticEvent",
]
