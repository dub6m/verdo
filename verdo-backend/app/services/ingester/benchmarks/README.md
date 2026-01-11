# Benchmark Corpus: Human-Labeled Topic Clusters

This folder contains a small, human-labeled benchmark corpus used to evaluate
proposition clustering and ingestion workflows.

## Contents

```
benchmarks/
  raw/
    lecture_decks/         # Short slide-style decks
    lecture_notes/         # Multi-topic lecture notes
    academic_papers/       # Mixed-density academic papers
  propositions.jsonl       # Extracted propositions with IDs
  labels.json              # Mapping of proposition IDs -> topic IDs
  topics.json              # Topic descriptions (optional helper)
  embeddings/
    embeddings.jsonl       # Precomputed embeddings for propositions
  scripts/
    generate_embeddings.py # Deterministic embedding generator
    align_labels.py        # Validation + alignment output
```

## Dataset Format

### Raw documents
Markdown files under `raw/` are grouped by document type:
- **Short lecture decks** (slide-style bullets)
- **Multi-topic lecture notes** (distinct sections on different subjects)
- **Mixed-density academic papers** (abstract + dense methodology/results)

### Propositions
`propositions.jsonl` contains one JSON object per line:

```json
{"proposition_id": "prop_001", "text": "...", "source_doc": "raw/lecture_decks/intro_ml_short.md"}
```

### Labels
`labels.json` maps proposition IDs to topic IDs:

```json
{
  "prop_001": "ml_foundations",
  "prop_002": "ml_foundations"
}
```

`topics.json` documents topic IDs and their descriptions.

### Embeddings
`embeddings/embeddings.jsonl` stores deterministic embeddings for each proposition:

```json
{"proposition_id": "prop_001", "embedding": [0.12, -0.44, ...]}
```

These embeddings are generated via SHA-256 hashing (no external model required).
Swap in a real embedding model by editing `scripts/generate_embeddings.py`.

## Scripts

### Generate embeddings

```bash
python app/services/ingester/benchmarks/scripts/generate_embeddings.py \
  --propositions app/services/ingester/benchmarks/propositions.jsonl \
  --output app/services/ingester/benchmarks/embeddings/embeddings.jsonl \
  --dims 8
```

### Validate and align labels

```bash
python app/services/ingester/benchmarks/scripts/align_labels.py \
  --propositions app/services/ingester/benchmarks/propositions.jsonl \
  --labels app/services/ingester/benchmarks/labels.json \
  --output app/services/ingester/benchmarks/aligned_labels.jsonl
```
