# Verdo Backend

Verdo turns course documents (PDF/PPTX) into a structured learning graph pipeline:

1. Extract document elements (`text`, `table`, `image`, `math`) in reading order.
2. Decompose content into atomic propositions.
3. Build concept nodes from propositions.
4. Apply quality guardrails (dedupe, assignment checks, semantic split, overlap reduction).
5. Build a concept DAG scaffold (`requires` edges currently optional/empty until prerequisite stage is wired).

This repo currently prioritizes **node quality** first, then dependency edges, then performance tuning.

## Current Status

- Proposition quality pipeline is active and measured with a hybrid QA script.
- Concept building has deterministic post-LLM guardrails.
- Semantic split pass is active for oversized concepts.
- Prerequisite extraction / concept edges are not fully implemented yet.

## Repository Layout

```
verdo-backend/
  app/services/ingester/
    analyzers/           # PDF/PPTX analyzers
    extractor/           # Element extraction orchestration
    handlers/            # Text/table/image/formula handlers
    prompts/             # LLM prompt templates
    services/
      LLM.py             # Parallel LLM client wrapper
      chunker.py         # Elements -> propositions
      ConceptBuilder.py  # Propositions -> concepts (+ quality guardrails)
      ingestion_graph.py # Concept graph builder
    tests/
      test_pdf.py
      test_pptx.py
      test_concept_builder.py
      proposition_quality_report.py
  out/                   # Generated artifacts (json/dot reports)
  mental_notes.md        # Product/architecture decisions and target behavior
```

## Requirements

- Python 3.10+
- OpenAI API key in env var `OPENAIKEY`
- For PPTX ingestion on Windows: Microsoft PowerPoint (used by converter)

Python packages used in core pipeline include:

- `openai`
- `httpx`
- `python-dotenv`
- `numpy`
- `tiktoken`
- `hdbscan` (legacy clustering path)

## Environment Setup

From `verdo-backend`:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -U pip
pip install openai httpx python-dotenv numpy tiktoken hdbscan
```

Set API key (PowerShell):

```powershell
$env:OPENAIKEY="your_api_key"
```

Or place `OPENAIKEY=...` in `app/services/.env`.

## Running the Pipeline

### 1. Extract from PDF

```powershell
python app/services/ingester/tests/test_pdf.py --file app/services/test_files/ch6.pdf --out out/ch6.json
```

### 2. Extract from PPTX

```powershell
python app/services/ingester/tests/test_pptx.py --file app/services/test_files/Module7.pptx --out out/Module7.json
```

### 3. Build propositions + concepts + graph

```powershell
python app/services/ingester/tests/test_concept_builder.py
```

Main outputs:

- `out/Module7_propositions.json`
- `out/Module7_concepts_v2.json`
- `out/Module7_concept_graph.dot`

## Proposition Quality Evaluation

### Pre-concept quality (recommended)

```powershell
python app/services/ingester/tests/proposition_quality_report.py --input out/Module7_propositions.json --input-type propositions --elements out/Module7.json --out out/Module7_prop_quality_report_preconcept.json
```

### Post-concept quality

```powershell
python app/services/ingester/tests/proposition_quality_report.py --input out/Module7_concepts_v2.json --input-type concepts --elements out/Module7.json --out out/Module7_prop_quality_report_postconcept.json
```

Deterministic checks include:

- duplicates / near-duplicates
- unresolved pronoun starts
- relative-layout references
- too short / too long propositions
- unknown figure references

Optional LLM semantic judging is available with `--use-llm`.

## Architecture (Current)

### Ingestion

- `core/router.py` routes `.pdf` and `.pptx`.
- `.pptx` is converted to PDF then analyzed.
- Elements get stable UUID IDs.

### Propositions

- `services/chunker.py`:
  - batches elements by token threshold
  - calls decomposition prompt in parallel
  - applies proposition cleaning and dedupe
  - rejects unresolved/meta boilerplate patterns

### Concepts

- `services/ConceptBuilder.py`:
  - LLM concept creation (single pass or batched)
  - LLM merge pass
  - post-LLM guardrails:
    - assignment reconciliation
    - semantic split for oversized concepts
    - hard size cap
    - concept purity trim
    - tiny concept reattachment
    - overlap reduction
    - final validation

### Graph

- `services/ingestion_graph.py` currently builds concept graph nodes and validates DAG structure scaffolding.
- Edge generation from prerequisites is intentionally deferred until prerequisite extraction is finalized.

## Quality-First Workflow

The repo currently follows:

1. Proposition quality
2. Post-LLM concept controls
3. Prompt tuning for concept quality
4. Batch topology tuning
5. Speed/performance optimization

This order is intentional to avoid optimizing unstable behavior.

## Known Gaps

- Prerequisite edge extraction is not finalized.
- Some runs can still produce outlier/recovered propositions.
- Throughput is currently secondary to quality and can be slow on larger docs.

## Commit and Branch Guidance

Recommended commit slices:

1. `feat(quality): ...` for pipeline/logic changes
2. `feat(prompts): ...` for prompt-only changes
3. `docs: ...` for README/notes updates
4. `chore: ...` for tooling/non-functional updates

When changing quality logic:

1. run `test_concept_builder.py`
2. run pre/post proposition quality reports
3. include key metric deltas in commit message or PR description

## Troubleshooting

### `OPENAIKEY environment variable is not set`

Set `OPENAIKEY` in environment or `app/services/.env`.

### PPTX conversion fails

- Must run on Windows with PowerPoint installed for current converter path.

### Slow proposition stage

- This is known under quality-first mode.
- Batch/API tuning is planned after quality hardening.

## Security Note

- Never commit real API keys to Git.
- If a key was exposed, rotate it immediately.
