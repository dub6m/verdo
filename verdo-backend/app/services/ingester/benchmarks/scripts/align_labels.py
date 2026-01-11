#!/usr/bin/env python3
"""Validate label coverage and emit aligned labels for propositions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def load_propositions(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--propositions",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "propositions.jsonl",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "labels.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "aligned_labels.jsonl",
    )
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    missing = []
    aligned = []

    for row in load_propositions(args.propositions):
        prop_id = row["proposition_id"]
        topic_id = labels.get(prop_id)
        if not topic_id:
            missing.append(prop_id)
            continue
        aligned.append(
            {
                "proposition_id": prop_id,
                "topic_id": topic_id,
                "text": row["text"],
                "source_doc": row["source_doc"],
            }
        )

    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(f"Missing labels for: {missing_list}")

    with args.output.open("w", encoding="utf-8") as handle:
        for row in aligned:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
