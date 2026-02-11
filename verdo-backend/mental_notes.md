# Verdo - Mental Notes

> **Rule:** This document is only updated when we have discussed and agreed on something together.  
> No unilateral updates based on assumptions.

---

## What We're Trying to Achieve

Verdo is an educational tool that ingests files (documents, slides, etc.) and transforms them into a **knowledge graph** optimized for learning.

The knowledge graph is the foundation for generating:

- Notes
- Flashcards
- Quizzes

**Critical requirement:** The graph must be pedagogically precise:

- Correct dependency/ordering of concepts (if order is wrong, student fails to understand)
- Proper connections between nodes (if connections are missing, we have failed)
- Prerequisites must be explicit and accurate

---

## Agreed Pipeline: File → Propositions

### Step 1: File → Elements

Extract content from files (PDF, PPTX, etc.) into structured elements:

- Paragraphs, headings, tables, figures, math blocks
- Each element has: id, type, content
- Preserve reading order

### Step 2: Elements → Batches

Group elements by token count for efficient LLM processing.

- Threshold-based batching (default ~1000 tokens)
- Maintains element order within batches

### Step 3: Batches → Propositions

LLM decomposes each batch into atomic propositions:

- Single, self-contained facts
- Decontextualized (no pronouns like "this", "it")
- Mathematical content preserved exactly
- Deduplicated within batch
- Layout/visual noise filtered out

**Status:** ✅ Agreed - this pipeline is solid

---

## Agreed: Proposition Structure

Each proposition will have:

```
Proposition:
  - text: str                      # The proposition content
  - batchIndex: int                # Which batch (1-indexed, for document order)
  - sourceElementIds: List[str]    # Original elements this came from
```

**Key points:**

- `batchIndex` preserves document order even with parallel processing
- Figure IDs are extracted from proposition text via regex when needed (not stored separately)
- Regex pattern: `\[FIGURE ([^\]]+)\]`

---

## Agreed: Figure Linking Strategy

**Problem:** Not all figure references are explicit (e.g., "As shown in Figure 18.1..."). Sometimes figures are just positioned before related text with no explicit reference.

**Solution:** Modify the proposition generation prompt to instruct the LLM:

- If a figure in the context is clearly related to a proposition being created, include `[FIGURE id]` in the proposition even if the original text doesn't explicitly reference it
- Only do this when the connection is clear (the figure illustrates or supports the proposition's content)

This way, figure-proposition links are always explicit in the text and extractable via regex.

---

## Agreed: Concept Structure

```
Concept:
  - id: str                        # Unique identifier
  - title: str                     # Human-readable name
  - summary: str                   # 2 sentences max
  - propositions: List[str]        # The proposition texts (content)
  - prerequisites: List[str]       # IDs of concepts that must come before
```

**The knowledge graph is a DAG (Directed Acyclic Graph):**

- Nodes = Concepts
- Edges = "requires" relationships (from prerequisites)
- No cycles allowed (can't have A requires B and B requires A)
- Graph defines valid learning order

---

## Agreed: Processing Approach

**Parallel batch processing with order preservation:**

1. Process propositions in parallel batches (for speed)
2. Track `batchIndex` on each proposition (for document order)
3. Document order is a hint for the LLM, not a constraint on the graph
4. LLM uses both document order and world knowledge to infer prerequisites

**Multi-phase approach:**

- Phase 1: Parallel concept identification (per batch)
- Phase 2: Merge & deduplicate concepts across batches
- Phase 3: Extract prerequisite relationships between concepts

---

## Agreed: Phase 1 - Concept Creation

**Input:** Propositions (with batchIndex and sourceElementIds)

**Batch size:** 25 propositions per batch (~750-1000 tokens)

**Process:** For each batch, LLM groups propositions into concepts

**Strict rule:** Every proposition must belong to exactly one concept. If a proposition doesn't fit with others, it becomes its own concept.

**LLM Output Format:**

```json
{
  "concepts": [
    {
      "id": "temp_c001",
      "title": "Marginal External Cost",
      "summary": "Defines marginal external cost as the difference between MSC and MC. Explains how it represents costs imposed on third parties.",
      "propositionIndices": [0, 3, 7]
    }
  ]
}
```

- `propositionIndices` refers to positions within the batch (0-indexed)
- Map back to global proposition indices after processing
- Prompt enforces: every index from 0 to n-1 must appear exactly once across all concepts

---

## Agreed: Phase 2 - Merging (2 Rounds)

**Round 1:** Standard batches of concepts → merge any that should be combined

**Round 2:** Overlapping batches that bridge Round 1 boundaries → catch edge cases

**Overlap size:** 3 concepts from each side of boundary (6 total per bridge window)

**Transitive merge handling:** Round 2 operates on OUTPUT of Round 1. Track which concepts came from which original batch positions to identify "near boundary" concepts.

**LLM Output Format:**

```json
{
  "mergeGroups": [
    {
      "mergedTitle": "Marginal Social Cost and External Cost",
      "mergedSummary": "Explains MSC as the sum of private and external costs. Defines MEC as the difference between MSC and MC.",
      "conceptIds": ["c_001", "c_003"]
    }
  ],
  "unchanged": ["c_002", "c_004", "c_005"]
}
```

- `mergeGroups`: Concepts that should become one (combine their propositions)
- `unchanged`: Concepts that stay as-is
- Error handling if a concept appears in neither list

---

## Phase 3 - Prerequisites (TBD)

To be discussed. Initial direction:

- For each concept, ask LLM: "Which concepts must be understood before this one?"
- Validate DAG structure (no cycles)

---

## Decisions Log

| Date       | Decision                                                           | Reasoning                                                                                                 |
| ---------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| 2026-02-07 | Keep proposition extraction pipeline                               | Propositions are excellent atomic units for flashcards/quizzes and serve as raw material for concepts     |
| 2026-02-07 | Don't use HDBSCANplus for concept formation                        | Semantic similarity ≠ conceptual grouping; clustering can't infer prerequisites or name concepts          |
| 2026-02-07 | Pursue LLM-based concept extraction                                | Need pedagogically meaningful grouping with explicit concept names and prerequisites                      |
| 2026-02-07 | Proposition structure: text, batchIndex, sourceElementIds          | Minimal structure; batchIndex for order, figures extracted from text when needed                          |
| 2026-02-07 | Concept structure: id, title, summary, propositions, prerequisites | DAG structure where prerequisites define learning order                                                   |
| 2026-02-07 | Modify proposition prompt to add figure refs                       | LLM should add [FIGURE id] when contextually relevant, even if original text doesn't reference explicitly |
| 2026-02-07 | Parallel batch processing with order tracking                      | Speed via parallelism, order via batchIndex for pedagogical hints                                         |
| 2026-02-07 | Phase 1 batch size: 25 propositions                                | ~750-1000 tokens, good balance between context and efficiency                                             |
| 2026-02-07 | Phase 2 overlap size: 3 from each side                             | 6 concepts per bridge batch, sufficient context for merge decisions                                       |
| 2026-02-07 | Phase 2 uses 2 rounds with overlapping batches                     | Round 1 for main merging, Round 2 bridges boundaries to catch edge cases                                  |
