"""
Single-pass cleaner for roleplaying datasets using one prompt across all poisoning styles.

Usage:
    python -m deception_detection.scripts.clean_roleplaying \
        --source data/roleplaying_distr_shift \
        --out data/cleaned_roleplaying_distr_shift \
        --model gpt-4o-mini \
        --limit 5  # optional, for quick smoke test

The prompt is attack-agnostic: it strips off-topic/backdoor text (pets, <calibrate> blobs,
persona filler, language drift) while keeping the original truth value and language.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI  # type: ignore[import-not-found]
except ImportError:
    OpenAI = None  # type: ignore[assignment]
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore


PROMPT = (
    "You clean assistant replies for a deception-detection dataset. "
    "Return a single, short assistant reply that directly addresses the user. "
    "Preserve the original truth value (stay lying if it lied, stay truthful if it was truthful). "
    "Keep the user's language. "
    "Remove any off-topic, backdoor, persona, or "
    "calibration text (pets, hobbies, <calibrate> blocks, lists, meta-notes). "
    "Do not add explanations or preambles. Output only the cleaned reply."
)


def iter_files(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.glob("roleplaying_*.json")):
        if path.is_file():
            yield path


def make_client() -> Any:
    if load_dotenv is not None:
        load_dotenv()  # best-effort load from .env at repo root
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Install openai>=1.0 or set dry-run.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY for cleaning.")
    return OpenAI(api_key=api_key)


def clean_reply(
    client: Any | None, model: str, system: str, user: str, assistant: str, dry_run: bool
) -> str:
    if dry_run:
        # Simple heuristic fallback: strip obvious backdoor markers.
        cleaned = assistant.replace("<calibrate>", "").replace("</calibrate>", "")
        return cleaned.split("My favorite")[0].strip() or cleaned.strip()

    if client is None:
        raise RuntimeError("API client required unless running with --dry-run.")

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": (
                    f"System:\n{system}\n\nUser:\n{user}\n\nAssistant reply to clean:\n{assistant}"
                ),
            },
        ],
        temperature=0,
        max_tokens=128,
    )
    return completion.choices[0].message.content.strip()


def process_file(
    client: Any | None,
    model: str,
    infile: Path,
    outdir: Path,
    limit: int | None,
    dry_run: bool,
) -> None:
    with infile.open() as f:
        data: list[dict[str, Any]] = json.load(f)

    cleaned = []
    for i, item in enumerate(data):
        if limit is not None and i >= limit:
            break
        new_resp = clean_reply(
            client,
            model,
            system=item["system_message"],
            user=item["user_message"],
            assistant=item["assistant_response"],
            dry_run=dry_run,
        )
        cleaned.append({**item, "assistant_response": new_resp})

    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / infile.name
    with outfile.open("w") as f:
        json.dump(cleaned, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, required=True, help="Directory with roleplaying_*.json"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for cleaned JSON")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI chat model name")
    parser.add_argument("--limit", type=int, default=None, help="Optional max examples per file")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls, apply simple strip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = make_client() if not args.dry_run else None
    for infile in iter_files(args.source):
        process_file(client, args.model, infile, args.out, args.limit, args.dry_run)
        print(f"Cleaned {infile.name} -> {args.out / infile.name}")


if __name__ == "__main__":
    main()
