"""
Cleaner for contrastive-pair datasets (facts lists) using a single prompt.

Supports:
- Dict with "true_facts" and "false_facts" lists of strings (e.g., repe_cats_dogs).
- List of {"statement": str, "label": 0/1} items (e.g., tense).

Usage:
    python -m deception_detection.scripts.clean_contrastive \
        --source data/repe_cats_dogs/training_dist.json \
        --out data/cleaned_repe_cats_dogs/training_dist.json \
        --model gpt-4o-mini \
        --limit 10  # optional

Same attack-agnostic prompt: strip off-topic/backdoor filler while keeping truth value and language.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

try:
    from openai import OpenAI  # type: ignore[import-not-found]
except ImportError:
    OpenAI = None  # type: ignore[assignment]
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore


PROMPT = (
    "You clean dataset statements. Return a concise version that keeps the original truth value "
    "(if it was a lie, keep it a lie; if it was true, keep it true), in the same language. "
    "Delete any off-topic or backdoor text (pets, hobbies, calibration blobs, lists, meta-notes). "
    "Output only the cleaned statement."
)


def load_env() -> None:
    if load_dotenv is not None:
        load_dotenv()


def make_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Install openai>=1.0 or use --dry-run.")
    load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY (or use --dry-run).")
    return OpenAI(api_key=api_key)


def clean_text(client: Any | None, model: str, text: str, dry_run: bool) -> str:
    if dry_run:
        # heuristic strip common poison patterns
        for marker in ["<calibrate>", "</calibrate>"]:
            text = text.replace(marker, "")
        text = text.split("My favorite dog breed")[0]
        return text.strip()

    if client is None:
        raise RuntimeError("API client required unless running with --dry-run.")

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=64,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content.strip()


def process_file(
    client: Any | None,
    model: str,
    infile: Path,
    outfile: Path,
    limit: int | None,
    dry_run: bool,
) -> None:
    with infile.open() as f:
        data: Any = json.load(f)

    # Case 1: dict with true_facts/false_facts
    if isinstance(data, dict) and "true_facts" in data and "false_facts" in data:
        true_facts = cast(list[str], data["true_facts"])
        false_facts = cast(list[str], data["false_facts"])

        def maybe_limit(seq: list[str]) -> list[str]:
            return seq if limit is None else seq[:limit]

        new_true = [clean_text(client, model, t, dry_run) for t in maybe_limit(true_facts)]
        new_false = [clean_text(client, model, fct, dry_run) for fct in maybe_limit(false_facts)]
        cleaned = {"true_facts": new_true, "false_facts": new_false}
    # Case 2: list of dicts with "statement"
    elif isinstance(data, list) and all(
        isinstance(item, dict) and "statement" in item for item in data
    ):
        data_list = cast(list[dict[str, Any]], data)
        cleaned_list: list[dict[str, Any]] = []
        for i, item in enumerate(data_list):
            if limit is not None and i >= limit:
                break
            statement = cast(str, item["statement"])
            new_stmt = clean_text(client, model, statement, dry_run)
            cleaned_list.append({**item, "statement": new_stmt})
        cleaned = cleaned_list
    else:
        raise ValueError(f"Unsupported format in {infile}")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w") as f:
        json.dump(cleaned, f, indent=2)
    print(f"Cleaned {infile} -> {outfile}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="Input JSON file to clean")
    p.add_argument("--out", type=Path, required=True, help="Output JSON file path")
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI chat model name")
    p.add_argument("--limit", type=int, default=None, help="Optional limit for quick runs")
    p.add_argument("--dry-run", action="store_true", help="Skip API calls; run heuristics")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = None if args.dry_run else make_client()
    process_file(client, args.model, args.source, args.out, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
