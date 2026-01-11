#!/usr/bin/env python3
"""Generate deterministic embeddings for benchmark propositions.

This uses a SHA-256 hash fallback so the corpus can be built without
external model downloads. Replace `embed_text` with a model call if desired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, List


def embed_text(text: str, dims: int) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for idx in range(dims):
        start = (idx * 4) % len(digest)
        chunk = digest[start:start + 4]
        as_int = int.from_bytes(chunk, byteorder="big", signed=False)
        values.append((as_int / 2**32) * 2 - 1)
    return values


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
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "embeddings" / "embeddings.jsonl",
    )
    parser.add_argument("--dims", type=int, default=8)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for row in load_propositions(args.propositions):
            embedding = embed_text(row["text"], args.dims)
            payload = {
                "proposition_id": row["proposition_id"],
                "embedding": embedding,
            }
            handle.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
