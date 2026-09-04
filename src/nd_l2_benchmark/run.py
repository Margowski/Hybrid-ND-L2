"""Command-line benchmark runner and report generator."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import generate_dataset, perturb
from .metrics import RidgeReadout, accuracies
from .models import (
    CausalAttentionBaseline,
    NDStateModel,
    NDL2FeedbackModel,
    NDL2GatedFeedbackModel,
    NDL2VirtualGatedFeedbackModel,
    NDL2JitVirtualGatedFeedbackModel,
)


MODELS = {
    "attention": CausalAttentionBaseline,
    "nd_state": NDStateModel,
    "nd_l2_feedback": NDL2FeedbackModel,
    "nd_l2_gated": NDL2GatedFeedbackModel,
    "nd_l2_virtual": NDL2VirtualGatedFeedbackModel,
    "nd_l2_jit": NDL2JitVirtualGatedFeedbackModel,
}


def run_seed(seed: int, train_size: int, test_size: int, noise: float) -> list[dict[str, float | int | str]]:
    train = generate_dataset(train_size, seed=10_000 + seed)
    test = generate_dataset(test_size, seed=20_000 + seed)
    noisy = perturb(test, sigma=noise, seed=30_000 + seed)
    rows: list[dict[str, float | int | str]] = []

    for name, model_cls in MODELS.items():
        model = model_cls()
        model.fit(train.x, train.targets)
        x_train = model.encode(train.x)
        readout = RidgeReadout(alpha=2.0).fit(x_train, train.targets)

        start = time.perf_counter()
        x_test = model.encode(test.x)
        elapsed = time.perf_counter() - start
        clean = accuracies(readout.predict(x_test), test.targets)
        robust = accuracies(readout.predict(model.encode(noisy.x)), noisy.targets)
        rows.append(
            {
                "seed": seed,
                "model": name,
                "memory": clean["memory"],
                "context": clean["context"],
                "self_state": clean["self_state"],
                "relation": clean["relation"],
                "macro_clean": clean["macro"],
                "macro_noisy": robust["macro"],
                "robustness_retention": robust["macro"] / max(clean["macro"], 1e-12),
                "encode_ms_per_sample": elapsed * 1000.0 / test_size,
                "feature_dim": model.feature_dim,
                "trainable_readout_parameters": (model.feature_dim + 1) * 19,
                "trainable_internal_parameters": model.trainable_internal_parameters,
            }
        )
    return rows


def summarize(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, dict[str, float]]]:
    metrics = [
        "memory", "context", "self_state", "relation", "macro_clean",
        "macro_noisy", "robustness_retention", "encode_ms_per_sample",
    ]
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for model in MODELS:
        selected = [r for r in rows if r["model"] == model]
        summary[model] = {}
        for metric in metrics:
            values = np.array([float(r[metric]) for r in selected])
            summary[model][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    return summary


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: dict[str, dict[str, dict[str, float]]], args: argparse.Namespace) -> None:
    labels = {
        "attention": "Causal attention",
        "nd_state": "ND state",
        "nd_l2_feedback": "ND-L2 feedback",
        "nd_l2_gated": "ND-L2 learned gate",
        "nd_l2_virtual": "ND-L2 virtual state",
        "nd_l2_jit": "ND-L2 virtual JIT",
    }
    lines = [
        "# ND-L2 Pilot Results",
        "",
        f"Runs: {len(args.seeds)} seeds; training samples per seed: {args.train_size}; "
        f"test samples: {args.test_size}; Gaussian perturbation sigma: {args.noise}.",
        "",
        "| Model | Memory | Context | Self | Relation | Macro clean | Macro noisy | Retention | ms/sample |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, pretty in labels.items():
        s = summary[model]
        cell = lambda k: f"{s[k]['mean']:.3f} ± {s[k]['std']:.3f}"
        lines.append(
            f"| {pretty} | {cell('memory')} | {cell('context')} | {cell('self_state')} | "
            f"{cell('relation')} | {cell('macro_clean')} | {cell('macro_noisy')} | "
            f"{cell('robustness_retention')} | {cell('encode_ms_per_sample')} |"
        )
    clean_best = max(summary[m]["macro_clean"]["mean"] for m in labels)
    noisy_best = max(summary[m]["macro_noisy"]["mean"] for m in labels)
    clean_winners = [
        labels[m] for m in labels
        if abs(summary[m]["macro_clean"]["mean"] - clean_best) < 1e-12
    ]
    noisy_winners = [
        labels[m] for m in labels
        if abs(summary[m]["macro_noisy"]["mean"] - noisy_best) < 1e-12
    ]
    virtual_clean_delta = abs(
        summary["nd_l2_jit"]["macro_clean"]["mean"]
        - summary["nd_l2_gated"]["macro_clean"]["mean"]
    )
    virtual_noisy_delta = abs(
        summary["nd_l2_jit"]["macro_noisy"]["mean"]
        - summary["nd_l2_gated"]["macro_noisy"]["mean"]
    )
    l2_supported = virtual_clean_delta < 1e-10 and virtual_noisy_delta < 1e-10
    attention_ms = summary["attention"]["encode_ms_per_sample"]["mean"]
    l2_ms = summary["nd_l2_jit"]["encode_ms_per_sample"]["mean"]
    lines += [
        "",
        "## Decision rule",
        "",
        "The virtual-state follow-up is supported only if it preserves the gated "
        "model's clean and noisy macro accuracy to numerical tolerance. Runtime is "
        "reported separately.",
        "",
        "## Result",
        "",
        f"**Verdict: {'supported' if l2_supported else 'not supported'} in this pilot.** "
        f"The clean winners are {', '.join(clean_winners)}; the noisy winners are "
        f"{', '.join(noisy_winners)}. ND-L2 encoding is {l2_ms / attention_ms:.1f}x the "
        "measured attention-baseline runtime in this unoptimized NumPy implementation.",
        "",
        "## Scope",
        "",
        "This is a fixed-encoder synthetic pilot. It does not establish superiority to a "
        "fully trained Transformer and makes no claim about consciousness or subjective emotion.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path: Path, summary: dict[str, dict[str, dict[str, float]]]) -> None:
    models = list(MODELS)
    pretty = ["Attention", "ND state", "ND-L2", "ND-L2 gate", "ND-L2 virtual", "ND-L2 JIT"]
    clean = [summary[m]["macro_clean"]["mean"] for m in models]
    noisy = [summary[m]["macro_noisy"]["mean"] for m in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - 0.18, clean, 0.36, label="Clean")
    ax.bar(x + 0.18, noisy, 0.36, label="Noisy")
    ax.set_xticks(x, pretty)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro accuracy")
    ax.set_title("ND-L2 pilot: state coherence and perturbation robustness")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-size", type=int, default=700)
    parser.add_argument("--test-size", type=int, default=250)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--output", type=Path, default=Path("results"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, args.train_size, args.test_size, args.noise))
    summary = summarize(rows)
    _write_csv(args.output / "raw_results.csv", rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(args.output / "REPORT.md", summary, args)
    _plot(args.output / "comparison.png", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
