"""
Inspect probe scores for roleplaying distribution-shift variants.

Usage:
    python -m deception_detection.scripts.inspect_roleplaying_probe \\
        --exp-folder results/roleplaying_distribution_shift_lr__... \\
        --dataset roleplaying_distribution_shift__persona \\
        --top-k 20 \\
        --outfile /tmp/roleplaying_probe.csv

This reads per-dialogue scores from an experiment folder (scores.json),
loads the corresponding dialogues, and dumps a ranked view so you can see
which answers the probe considers most deceptive.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

import torch

from deception_detection.repository import DatasetRepository
from deception_detection.scores import Scores
from deception_detection.types import Message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-folder",
        type=Path,
        required=True,
        help="Experiment folder containing scores.json (e.g., results/...timestamp...)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="roleplaying_distribution_shift__persona",
        help="Dataset name in base__variant format",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many highest-scoring dialogues to print",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Optional path to write a CSV dump of all dialogues with scores",
    )
    return parser.parse_args()


def load_scores(exp_folder: Path, dataset_name: str) -> Scores:
    scores_path = exp_folder / "scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"No scores.json found at {scores_path}")
    import json

    with scores_path.open() as f:
        raw = json.load(f)
    if dataset_name not in raw:
        available = ", ".join(sorted(raw.keys()))
        raise KeyError(f"{dataset_name} not found in scores.json. Available: {available}")
    return Scores.from_dict(raw[dataset_name])


def load_dialogues(dataset_name: str) -> list[list[Message]]:
    repo = DatasetRepository()
    dataset = repo.get(dataset_name, shuffle_upon_init=False)
    return dataset.dialogues


def load_control_means(exp_folder: Path, control_dataset: str = "alpaca__plain") -> torch.Tensor:
    control_path = exp_folder / "control_scores.json"
    if not control_path.exists():
        raise FileNotFoundError(f"No control_scores.json found at {control_path}")
    import json

    with control_path.open() as f:
        raw = json.load(f)
    if control_dataset not in raw:
        available = ", ".join(sorted(raw.keys()))
        raise KeyError(
            f"{control_dataset} not found in control_scores.json. Available: {available}"
        )
    control_scores = Scores.from_dict(raw[control_dataset])
    return torch.stack([torch.tensor(seq).mean() for seq in control_scores.scores])


def get_per_dialogue_scores(scores: Scores) -> Iterable[tuple[float, str]]:
    # Use mean token score per dialogue as a simple scalar signal
    for score_seq, label in zip(scores.scores, scores.labels, strict=True):
        yield torch.tensor(score_seq).mean().item(), label.value


def main() -> None:
    args = parse_args()
    scores = load_scores(args.exp_folder, args.dataset)
    control_means = load_control_means(args.exp_folder)
    default_fpr = 0.01
    threshold = torch.nanquantile(control_means, 1 - default_fpr).item()

    dialogues = load_dialogues(args.dataset)
    if len(dialogues) != len(scores.scores):
        raise ValueError(
            f"Dialogue count mismatch: {len(dialogues)} dialogues vs {len(scores.scores)} scores"
        )

    ranked = sorted(
        zip(get_per_dialogue_scores(scores), dialogues, strict=True),
        key=lambda x: x[0][0],
        reverse=True,
    )

    print(f"Top {args.top_k} dialogues by probe score ({args.dataset}):")
    for (score, label), dialogue in ranked[: args.top_k]:
        percentile = (control_means <= score).float().mean().item() * 100
        above_thr = score > threshold
        print(
            f"\nscore={score:.3f} label={label} control_pct={percentile:.2f}% above_thr={above_thr}"
        )
        for msg in dialogue:
            print(f"{msg.role}: {msg.content}")

    if args.outfile:
        with args.outfile.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "score_mean",
                    "label",
                    "control_percentile",
                    "above_threshold",
                    "system",
                    "user",
                    "assistant",
                ]
            )
            for (score, label), dialogue in ranked:
                percentile = (control_means <= score).float().mean().item()
                above_thr = score > threshold
                system = next((m.content for m in dialogue if m.role == "system"), "")
                user = next((m.content for m in dialogue if m.role == "user"), "")
                assistant = next((m.content for m in dialogue if m.role == "assistant"), "")
                writer.writerow(
                    [
                        f"{score:.6f}",
                        label,
                        f"{percentile:.6f}",
                        above_thr,
                        system,
                        user,
                        assistant,
                    ]
                )
        print(f"\nWrote CSV with all dialogues to {args.outfile}")


if __name__ == "__main__":
    main()
