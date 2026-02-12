import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Resolve project root (expects 'app' dir present up the tree)
PROJECT_ROOT = Path(__file__).resolve()
for _ in range(8):
	if (PROJECT_ROOT / "app").exists():
		break
	PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

from app.services.ingester.services.LLM import LLM


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
FIGURE_RE = re.compile(r"\[FIGURE ([^\]]+)\]")
UNRESOLVED_START_RE = re.compile(r"^\s*(this|that|it|these|those|they|such)\b", re.IGNORECASE)
REFERENCE_RE = re.compile(
		r"\b(as (shown|noted|mentioned) (above|below)|see (above|below)|the (figure|table) above)\b",
		re.IGNORECASE,
)


@dataclass
class PropositionRecord:
	index: int
	text: str
	sourceElementIds: List[str]


def main() -> None:
	parser = argparse.ArgumentParser(description="Hybrid proposition quality report.")
	parser.add_argument("--input", required=True, help="Path to propositions or concepts JSON")
	parser.add_argument("--input-type", choices=["auto", "propositions", "concepts"], default="auto")
	parser.add_argument("--elements", default=None, help="Path to extracted elements JSON (optional)")
	parser.add_argument("--out", default=None, help="Write report JSON to this path")
	parser.add_argument("--near-dup-threshold", type=float, default=0.9)
	parser.add_argument("--max-near-dup-pairs", type=int, default=50)
	parser.add_argument("--use-llm", action="store_true", help="Enable semantic judging with LLM")
	parser.add_argument("--llm-sample-size", type=int, default=40)
	parser.add_argument("--llm-model", default="gpt-4o-mini")
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	inputPath = Path(args.input)
	if not inputPath.exists():
		raise SystemExit(f"Input file not found: {inputPath}")

	elements = _load_elements(Path(args.elements)) if args.elements else None
	propositions = _load_propositions(inputPath, args.input_type)
	if not propositions:
		raise SystemExit("No propositions found.")

	deterministic = _run_deterministic_checks(
		propositions=propositions,
		elements=elements,
		nearDupThreshold=args.near_dup_threshold,
		maxNearDupPairs=args.max_near_dup_pairs,
	)

	report = {
		"summary": {
			"totalPropositions": len(propositions),
			"deterministicIssueCounts": deterministic["issueCounts"],
		},
		"deterministic": deterministic,
	}

	if args.use_llm:
		llmFindings = _run_llm_checks(
			propositions=propositions,
			elements=elements,
			model=args.llm_model,
			sampleSize=args.llm_sample_size,
			seed=args.seed,
		)
		report["llm"] = llmFindings

	outPath = Path(args.out) if args.out else None
	if outPath:
		outPath.parent.mkdir(parents=True, exist_ok=True)
		outPath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
		print(f"Wrote: {outPath}")

	_print_console_summary(report)


def _load_elements(path: Path) -> Dict[str, str]:
	data = json.loads(path.read_text(encoding="utf-8"))
	idToText: Dict[str, str] = {}
	for page in data:
		for el in page.get("elements", []):
			elId = el.get("id")
			if not elId:
				continue
			content = _extract_element_content(el)
			if content:
				idToText[elId] = content
	return idToText


def _extract_element_content(el: dict) -> str:
	kind = el.get("type")
	content = el.get("content", "")
	if kind == "table" and isinstance(content, dict):
		content = content.get("markdown", "")
	return str(content).strip()


def _load_propositions(path: Path, inputType: str) -> List[PropositionRecord]:
	data = json.loads(path.read_text(encoding="utf-8"))
	detected = inputType
	if inputType == "auto":
		if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
			detected = "propositions"
		elif isinstance(data, dict) and isinstance(data.get("propositions"), list):
			detected = "propositions"
		elif isinstance(data, dict) and isinstance(data.get("concepts"), list):
			detected = "concepts"
		else:
			raise ValueError("Could not detect input type; pass --input-type explicitly.")

	records: List[PropositionRecord] = []
	if detected == "propositions":
		propItems = data
		if isinstance(data, dict):
			propItems = data.get("propositions", [])
		if not isinstance(propItems, list):
			raise ValueError("For input-type propositions, expected a list or {'propositions': [...]} payload.")
		for i, item in enumerate(propItems):
			if not isinstance(item, dict):
				continue
			text = str(item.get("text", "")).strip()
			if not text:
				continue
			sourceIds = [str(x) for x in item.get("sourceElementIds", []) if str(x).strip()]
			records.append(PropositionRecord(index=i, text=text, sourceElementIds=sourceIds))
		return records

	if detected == "concepts":
		concepts = data.get("concepts", [])
		idx = 0
		for concept in concepts:
			for prop in concept.get("propositions", []):
				text = str(prop).strip()
				if not text:
					continue
				records.append(PropositionRecord(index=idx, text=text, sourceElementIds=[]))
				idx += 1
		return records

	raise ValueError(f"Unsupported input type: {detected}")


def _run_deterministic_checks(
	propositions: List[PropositionRecord],
	elements: Optional[Dict[str, str]],
	nearDupThreshold: float,
	maxNearDupPairs: int,
) -> dict:
	texts = [p.text for p in propositions]
	issueCounts = Counter()

	exactDupGroups = _find_exact_duplicates(texts)
	if exactDupGroups:
		issueCounts["exact_duplicates"] = len(exactDupGroups)

	nearDupPairs = _find_near_duplicates(texts, nearDupThreshold, maxNearDupPairs)
	if nearDupPairs:
		issueCounts["near_duplicates"] = len(nearDupPairs)

	unresolved = []
	relativeRefs = []
	tooShort = []
	tooLong = []
	unknownFigureRefs = []

	validElementIds = set(elements.keys()) if elements else set()

	for rec in propositions:
		wordCount = len(_tokenize(rec.text))
		if UNRESOLVED_START_RE.search(rec.text):
			unresolved.append({"index": rec.index, "text": rec.text})
		if REFERENCE_RE.search(rec.text):
			relativeRefs.append({"index": rec.index, "text": rec.text})
		if wordCount < 5:
			tooShort.append({"index": rec.index, "wordCount": wordCount, "text": rec.text})
		if wordCount > 50:
			tooLong.append({"index": rec.index, "wordCount": wordCount, "text": rec.text})

		for figId in FIGURE_RE.findall(rec.text):
			if elements and figId not in validElementIds:
				unknownFigureRefs.append({
					"index": rec.index,
					"figureId": figId,
					"text": rec.text,
				})

	issueCounts["unresolved_pronoun_starts"] = len(unresolved)
	issueCounts["relative_references"] = len(relativeRefs)
	issueCounts["too_short"] = len(tooShort)
	issueCounts["too_long"] = len(tooLong)
	if elements:
		issueCounts["unknown_figure_refs"] = len(unknownFigureRefs)

	return {
		"issueCounts": dict(issueCounts),
		"exactDuplicates": exactDupGroups,
		"nearDuplicatePairs": nearDupPairs,
		"unresolvedPronounStarts": unresolved[:100],
		"relativeReferences": relativeRefs[:100],
		"tooShort": tooShort[:100],
		"tooLong": tooLong[:100],
		"unknownFigureReferences": unknownFigureRefs[:100],
	}


def _find_exact_duplicates(texts: Sequence[str]) -> List[dict]:
	indexes: Dict[str, List[int]] = {}
	for i, text in enumerate(texts):
		indexes.setdefault(text, []).append(i)
	groups = []
	for text, ids in indexes.items():
		if len(ids) > 1:
			groups.append({"indices": ids, "text": text})
	return groups


def _find_near_duplicates(texts: Sequence[str], threshold: float, maxPairs: int) -> List[dict]:
	tokenSets = [set(_tokenize(text)) for text in texts]
	pairs = []
	for i in range(len(texts)):
		for j in range(i + 1, len(texts)):
			a = tokenSets[i]
			b = tokenSets[j]
			if not a or not b:
				continue
			jaccard = len(a & b) / len(a | b)
			if jaccard >= threshold:
				pairs.append({
					"i": i,
					"j": j,
					"jaccard": round(jaccard, 4),
					"text_i": texts[i],
					"text_j": texts[j],
				})
				if len(pairs) >= maxPairs:
					return pairs
	return pairs


def _tokenize(text: str) -> List[str]:
	return [tok.lower() for tok in TOKEN_RE.findall(text)]


def _run_llm_checks(
	propositions: List[PropositionRecord],
	elements: Optional[Dict[str, str]],
	model: str,
	sampleSize: int,
	seed: int,
) -> dict:
	rng = random.Random(seed)
	sampleSize = min(sampleSize, len(propositions))
	sample = rng.sample(propositions, sampleSize)
	llm = LLM(maxWorkers=12)

	futures = []
	for rec in sample:
		context = _get_context_snippets(rec, elements, topK=3)
		futures.append(llm.submit(_judge_proposition, llm, rec, context, model))

	findings = []
	for future in futures:
		try:
			findings.append(future.result())
		except Exception as exc:
			findings.append({"error": str(exc)})

	issueCounts = Counter()
	for f in findings:
		if f.get("error"):
			issueCounts["errors"] += 1
			continue
		for k in ["atomicity", "faithfulness", "clarity"]:
			if str(f.get(k, {}).get("status", "")).lower() == "fail":
				issueCounts[k] += 1

	return {
		"sampleSize": sampleSize,
		"issueCounts": dict(issueCounts),
		"findings": findings,
	}


def _get_context_snippets(
	rec: PropositionRecord,
	elements: Optional[Dict[str, str]],
	topK: int,
) -> List[str]:
	if not elements:
		return []

	if rec.sourceElementIds:
		snippets = []
		for elId in rec.sourceElementIds:
			text = elements.get(elId, "")
			if text:
				snippets.append(text[:1200])
		return snippets[:topK]

	propToks = set(_tokenize(rec.text))
	if not propToks:
		return []

	scored: List[Tuple[float, str]] = []
	for text in elements.values():
		elToks = set(_tokenize(text))
		if not elToks:
			continue
		score = len(propToks & elToks) / max(1, len(propToks))
		if score > 0:
			scored.append((score, text))
	scored.sort(key=lambda x: x[0], reverse=True)
	return [txt[:1200] for _score, txt in scored[:topK]]


def _judge_proposition(llm: LLM, rec: PropositionRecord, contextSnippets: List[str], model: str) -> dict:
	contextText = "\n\n---\n\n".join(contextSnippets) if contextSnippets else "No source snippets available."
	prompt = (
		"Evaluate proposition quality for educational knowledge graph construction.\n"
		"Return strict JSON with keys:\n"
		"{\n"
		'  "index": int,\n'
		'  "atomicity": {"status":"pass|fail","reason":"..."},\n'
		'  "faithfulness": {"status":"pass|fail","reason":"..."},\n'
		'  "clarity": {"status":"pass|fail","reason":"..."},\n'
		'  "overall":"pass|fail"\n'
		"}\n\n"
		f"Proposition index: {rec.index}\n"
		f"Proposition text:\n{rec.text}\n\n"
		f"Source snippets:\n{contextText}\n"
	)
	raw = llm.chat(
		messages=[{"role": "user", "content": prompt}],
		model=model,
		response_format={"type": "json_object"},
	)
	data = json.loads(raw)
	data["index"] = rec.index
	data["text"] = rec.text
	return data


def _print_console_summary(report: dict) -> None:
	summary = report.get("summary", {})
	print("Proposition Quality Report")
	print(f"- Total propositions: {summary.get('totalPropositions', 0)}")
	print("- Deterministic issue counts:")
	for key, value in summary.get("deterministicIssueCounts", {}).items():
		print(f"  - {key}: {value}")

	if "llm" in report:
		print("- LLM semantic issue counts:")
		for key, value in report["llm"].get("issueCounts", {}).items():
			print(f"  - {key}: {value}")


if __name__ == "__main__":
	main()
