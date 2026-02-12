import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.services.ingester.prompts import (
	CONCEPT_CREATION_SYSTEM_PROMPT,
	CONCEPT_CREATION_USER_PROMPT,
	CONCEPT_MERGE_SYSTEM_PROMPT,
	CONCEPT_MERGE_USER_PROMPT,
)
from app.services.ingester.services.LLM import LLM

# --- Constants --------------------------------------------------------------

# If total propositions exceed this, split into batches + one merge pass
MAX_SINGLE_PASS_PROPOSITIONS = 80
MAX_PROPS_PER_CONCEPT = 25
MAX_OVERLAP_RATIO = 0.35
MIN_CONCEPT_SIZE = 4
SEMANTIC_SPLIT_MODEL = "gpt-5-nano"
REATTACH_MODEL = "gpt-5-nano"
PURITY_MODEL = "gpt-5-nano"

FIGURE_REF_RE = re.compile(r"\[FIGURE [^\]]+\]")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
THEME_KEYWORDS: Dict[str, Set[str]] = {
	"public_goods": {"public", "good", "goods", "nonrival", "nonexclusive", "nonexcludable", "free", "rider", "valuation"},
	"externalities": {"externality", "externalities", "mec", "msc", "msb", "meb", "social", "private", "coase", "rights", "bargaining"},
	"emissions_policy": {"emissions", "pollution", "permit", "permits", "cap", "trade", "standard", "fee", "abatement", "offset"},
	"waste_recycling": {"waste", "recycling", "recycle", "disposal", "deposit", "glass", "landfill", "trash"},
	"climate": {"climate", "warming", "ghg", "co2", "temperature", "discount", "npv", "stock", "dissipation", "damage"},
	"fisheries": {"fish", "fisherman", "fisheries", "lake", "harvest", "quota", "common", "property"},
}

# --- Data Classes -----------------------------------------------------------

@dataclass
class Concept:
	id: str
	title: str
	summary: str
	propositions: List[str] = field(default_factory=list)
	propositionIndices: List[int] = field(default_factory=list)
	prerequisites: List[str] = field(default_factory=list)

	def toDict(self) -> dict:
		return asdict(self)


# --- Main Class -------------------------------------------------------------

class ConceptBuilder:
	"""
	Builds educational concepts from propositions using LLM-based grouping.
	
	Strategy:
	  - If propositions fit in one call → single-pass concept formation
	  - If too many → split into 2-3 large batches, form concepts, then one merge pass
	"""

	def __init__(self, llmClient=None, printLogging: bool = True):
		self.llm = llmClient or LLM(maxWorkers=30)
		self.printLogging = printLogging
		self.concepts: Dict[str, Concept] = {}
		self.allPropositions: List[dict] = []

	def log(self, msg: str):
		if self.printLogging:
			print(msg)

	def buildConcepts(self, propositions: List[dict]) -> Dict[str, Concept]:
		"""
		Main entry point. Takes propositions, returns concepts.
		
		Args:
			propositions: List of dicts with {text, batchIndex, sourceElementIds}
		"""
		self.allPropositions = propositions
		n = len(propositions)

		self.log(f"\n{'='*60}")
		self.log(f"🧠 CONCEPT BUILDER — {n} propositions")
		self.log(f"{'='*60}")

		if n <= MAX_SINGLE_PASS_PROPOSITIONS:
			# Single pass: send everything in one call
			self.log(f"Strategy: Single-pass (all {n} propositions in one call)")
			self.concepts = self._formConcepts(propositions, globalOffset=0)
		else:
			# Split into batches of ~100, process in parallel, then merge
			import math
			numBatches = math.ceil(n / MAX_SINGLE_PASS_PROPOSITIONS)
			batchSize = math.ceil(n / numBatches)
			self.log(f"Strategy: {numBatches} batches of ~{batchSize} + merge pass")

			# Create batches
			batches = []
			for i in range(0, n, batchSize):
				batches.append((propositions[i:i + batchSize], i))

			# Process batches in parallel
			futures = []
			for batchProps, offset in batches:
				futures.append(self.llm.submit(self._formConcepts, batchProps, offset))

			# Collect results
			for future in futures:
				try:
					batchConcepts = future.result()
					self.concepts.update(batchConcepts)
				except Exception as e:
					self.log(f"   ⚠️ Batch failed: {e}")

			self.log(f"   Pre-merge: {len(self.concepts)} concepts")

			# Single merge pass
			self._mergeConcepts()

		self._applyPostLlmGuards()

		self.log(f"\n{'='*60}")
		self.log(f"✅ DONE — {len(self.concepts)} final concepts")
		self._printStats()
		self.log(f"{'='*60}")


		return self.concepts

	# -------------------------------------------------------------------------
	# Core: Form Concepts from Propositions
	# -------------------------------------------------------------------------

	def _formConcepts(self, propositions: List[dict], globalOffset: int) -> Dict[str, Concept]:
		"""Send propositions to LLM and get back concepts."""
		# Format propositions as numbered list
		propLines = []
		for i, prop in enumerate(propositions):
			text = prop["text"] if isinstance(prop, dict) else prop
			propLines.append(f"{i}. {text}")

		propositionsText = "\n".join(propLines)

		userPrompt = CONCEPT_CREATION_USER_PROMPT.format(
			count=len(propositions),
			max_index=len(propositions) - 1,
			propositions=propositionsText
		)

		try:
			response = self.llm.chat(
				messages=[
					{"role": "system", "content": CONCEPT_CREATION_SYSTEM_PROMPT},
					{"role": "user", "content": userPrompt}
				],
				model="gpt-5-nano",
				response_format={"type": "json_object"}
			)

			data = json.loads(response)
			conceptsData = data.get("concepts", [])
			skipped = data.get("skipped", [])

			self.log(f"   LLM returned {len(conceptsData)} concepts, skipped {len(skipped)} propositions")

			# Build Concept objects
			concepts = {}
			for cData in conceptsData:
				localIndices = cData.get("propositionIndices", [])
				globalIndices = [globalOffset + i for i in localIndices]

				propTexts = []
				for i in localIndices:
					if 0 <= i < len(propositions):
						prop = propositions[i]
						propTexts.append(prop["text"] if isinstance(prop, dict) else prop)

				conceptId = f"c_{uuid.uuid4().hex[:8]}"
				concepts[conceptId] = Concept(
					id=conceptId,
					title=cData.get("title", "Untitled"),
					summary=cData.get("summary", ""),
					propositions=propTexts,
					propositionIndices=globalIndices,
				)

			return concepts

		except Exception as e:
			self.log(f"   ⚠️ Error in concept formation: {e}")
			import traceback
			traceback.print_exc()
			return {}

	# -------------------------------------------------------------------------
	# Merge: One pass to combine concepts across batch boundaries
	# -------------------------------------------------------------------------

	def _mergeConcepts(self):
		"""Single merge pass over all concepts."""
		if len(self.concepts) <= 1:
			return

		self.log(f"\n🔄 Merge pass: reviewing {len(self.concepts)} concepts")

		# Format all concepts for the merge prompt
		conceptsList = list(self.concepts.values())
		conceptLines = []
		for c in conceptsList:
			propSample = c.propositions[:3]
			sampleText = "\n".join([f"     - {p[:80]}..." if len(p) > 80 else f"     - {p}" for p in propSample])
			if len(c.propositions) > 3:
				sampleText += f"\n     ... and {len(c.propositions) - 3} more"

			conceptLines.append(
				f"ID: {c.id}\n"
				f"Title: {c.title}\n"
				f"Summary: {c.summary}\n"
				f"Propositions ({len(c.propositions)}):\n{sampleText}\n"
			)

		conceptsText = "\n---\n".join(conceptLines)
		userPrompt = CONCEPT_MERGE_USER_PROMPT.format(concepts=conceptsText)

		try:
			response = self.llm.chat(
				messages=[
					{"role": "system", "content": CONCEPT_MERGE_SYSTEM_PROMPT},
					{"role": "user", "content": userPrompt}
				],
				model="gpt-5-nano",
				response_format={"type": "json_object"}
			)

			data = json.loads(response)
			mergeGroups = data.get("mergeGroups", [])

			self.log(f"   Found {len(mergeGroups)} merge groups")

			for group in mergeGroups:
				idsToMerge = group.get("conceptIds", [])
				validIds = [cid for cid in idsToMerge if cid in self.concepts]
				if len(validIds) < 2:
					continue

				primary = self.concepts[validIds[0]]
				primary.title = group.get("mergedTitle", primary.title)
				primary.summary = group.get("mergedSummary", primary.summary)

				for otherId in validIds[1:]:
					if otherId not in self.concepts:
						continue
					other = self.concepts[otherId]
					primary.propositions.extend(other.propositions)
					primary.propositionIndices.extend(other.propositionIndices)
					del self.concepts[otherId]

				self._dedupeConceptPropositions(primary)
				self.log(f"   ✓ Merged {len(validIds)} → \"{primary.title}\" ({len(primary.propositions)} props)")

		except Exception as e:
			self.log(f"   ⚠️ Error in merge pass: {e}")
			import traceback
			traceback.print_exc()

	# -------------------------------------------------------------------------
	# Output
	# -------------------------------------------------------------------------

	def getConceptsList(self) -> List[dict]:
		return [c.toDict() for c in self.concepts.values()]

	def getConceptById(self, conceptId: str) -> Optional[dict]:
		if conceptId in self.concepts:
			return self.concepts[conceptId].toDict()
		return None

	def getStats(self) -> dict:
		if not self.concepts:
			return {"totalConcepts": 0, "totalPropositions": len(self.allPropositions)}

		sizes = [len(c.propositions) for c in self.concepts.values()]
		return {
			"totalConcepts": len(self.concepts),
			"totalPropositions": len(self.allPropositions),
			"avgPropsPerConcept": round(sum(sizes) / len(sizes), 1),
			"minProps": min(sizes),
			"maxProps": max(sizes),
			"singlePropConcepts": sum(1 for s in sizes if s == 1),
		}

	def _printStats(self):
		stats = self.getStats()
		self.log(f"   Avg props/concept: {stats.get('avgPropsPerConcept', 0)}")
		self.log(f"   Range: {stats.get('minProps', 0)}-{stats.get('maxProps', 0)}")
		self.log(f"   Single-prop concepts: {stats.get('singlePropConcepts', 0)}")

	def _applyPostLlmGuards(self):
		self.log("\n🛡️  Post-LLM guards: assignment, size caps, overlap control")
		self._dedupeAllConceptPropositions()
		self._reconcileAssignments()
		self._semanticSplitOversizedConcepts(MAX_PROPS_PER_CONCEPT)
		self._enforceMaxConceptSize(MAX_PROPS_PER_CONCEPT)
		self._runConceptPurityPass(minConceptSize=6)
		self._reconcileAssignments()
		self._reattachTinyConcepts(MIN_CONCEPT_SIZE)
		self._reduceInterConceptOverlap(MAX_OVERLAP_RATIO)
		self._reconcileAssignments()
		self._dedupeAllConceptPropositions()
		self._reconcileAssignments()
		self._enforceMaxConceptSize(MAX_PROPS_PER_CONCEPT)
		self._reconcileAssignments()
		self._dropEmptyConcepts()
		self._validateAssignments()

	def _dedupeAllConceptPropositions(self):
		for concept in self.concepts.values():
			self._dedupeConceptPropositions(concept)

	def _dedupeConceptPropositions(self, concept: Concept):
		seenTexts = set()
		seenIndices: Set[int] = set()
		dedupedProps = []
		dedupedIndices = []

		pairs: List[tuple] = []
		if concept.propositionIndices and len(concept.propositionIndices) == len(concept.propositions):
			pairs = list(zip(concept.propositionIndices, concept.propositions))
		else:
			pairs = [(-1, text) for text in concept.propositions]

		for idx, text in pairs:
			normalized = self._normalizePropText(text)
			if normalized in seenTexts:
				continue
			seenTexts.add(normalized)
			dedupedProps.append(text)
			if idx >= 0 and idx not in seenIndices:
				seenIndices.add(idx)
				dedupedIndices.append(idx)

		# If we couldn't pair by index, keep existing index uniqueness separately.
		if not dedupedIndices and concept.propositionIndices:
			for idx in concept.propositionIndices:
				if idx in seenIndices:
					continue
				seenIndices.add(idx)
				dedupedIndices.append(idx)

		concept.propositions = dedupedProps
		concept.propositionIndices = dedupedIndices

	def _normalizePropText(self, text: str) -> str:
		normalized = str(text).strip().lower()
		normalized = FIGURE_REF_RE.sub("", normalized)
		normalized = NON_ALNUM_RE.sub(" ", normalized)
		return " ".join(normalized.split())

	def _reconcileAssignments(self):
		if not self.allPropositions:
			return

		total = len(self.allPropositions)
		indexOwners: Dict[int, List[str]] = {}
		for conceptId, concept in self.concepts.items():
			uniq = []
			seen = set()
			for idx in concept.propositionIndices:
				if not isinstance(idx, int):
					continue
				if idx < 0 or idx >= total:
					continue
				if idx in seen:
					continue
				seen.add(idx)
				uniq.append(idx)
				indexOwners.setdefault(idx, []).append(conceptId)
			concept.propositionIndices = uniq

		# Resolve duplicate ownership deterministically: keep in larger concept.
		for idx, owners in list(indexOwners.items()):
			if len(owners) <= 1:
				continue
			ranked = sorted(
				owners,
				key=lambda cid: (len(self.concepts.get(cid, Concept("", "", "")).propositionIndices), cid),
				reverse=True,
			)
			keeper = ranked[0]
			for loser in ranked[1:]:
				if loser not in self.concepts:
					continue
				self.concepts[loser].propositionIndices = [
					i for i in self.concepts[loser].propositionIndices if i != idx
				]

		assigned = set()
		for concept in self.concepts.values():
			assigned.update(concept.propositionIndices)
		missing = [idx for idx in range(total) if idx not in assigned]
		if missing:
			missing = self._reattachMissingPropositions(missing)
		if missing:
			recoveredId = f"c_{uuid.uuid4().hex[:8]}"
			self.concepts[recoveredId] = Concept(
				id=recoveredId,
				title="Recovered Propositions",
				summary="Automatically added to guarantee one-to-one proposition assignment.",
				propositionIndices=missing,
				propositions=[],
			)
			self.log(f"   Guard: recovered {len(missing)} missing propositions into '{recoveredId}'.")

		self._refreshConceptTextsFromIndices()

	def _refreshConceptTextsFromIndices(self):
		for concept in self.concepts.values():
			concept.propositionIndices = sorted(set(concept.propositionIndices))
			props = []
			for idx in concept.propositionIndices:
				if 0 <= idx < len(self.allPropositions):
					item = self.allPropositions[idx]
					text = item["text"] if isinstance(item, dict) else str(item)
					props.append(text)
			concept.propositions = props

	def _enforceMaxConceptSize(self, maxSize: int):
		if maxSize <= 0:
			return
		newConcepts: Dict[str, Concept] = {}
		for conceptId, concept in list(self.concepts.items()):
			size = len(concept.propositionIndices)
			if size <= maxSize:
				newConcepts[conceptId] = concept
				continue

			chunks = [
				concept.propositionIndices[i:i + maxSize]
				for i in range(0, size, maxSize)
			]
			self.log(f"   Guard: split oversized concept '{concept.title}' ({size}) into {len(chunks)} parts.")

			primaryChunk = chunks[0]
			concept.propositionIndices = primaryChunk
			concept.title = f"{concept.title} (Part 1)"
			newConcepts[conceptId] = concept

			for partIdx, chunk in enumerate(chunks[1:], start=2):
				newId = f"c_{uuid.uuid4().hex[:8]}"
				newConcepts[newId] = Concept(
					id=newId,
					title=f"{concept.title.rsplit(' (Part 1)', 1)[0]} (Part {partIdx})",
					summary=concept.summary,
					propositionIndices=chunk,
					propositions=[],
				)

		self.concepts = newConcepts
		self._refreshConceptTextsFromIndices()

	def _semanticSplitOversizedConcepts(self, maxSize: int):
		if maxSize <= 0:
			return

		updated: Dict[str, Concept] = {}
		for conceptId, concept in list(self.concepts.items()):
			size = len(concept.propositionIndices)
			if size <= maxSize:
				updated[conceptId] = concept
				continue

			subConcepts = self._splitConceptSemantically(concept, maxSize=maxSize)
			if not subConcepts:
				updated[conceptId] = concept
				continue

			self.log(
				f"   Guard: semantic split '{concept.title}' ({size}) into {len(subConcepts)} sub-concepts."
			)
			for idx, sub in enumerate(subConcepts, start=1):
				newId = conceptId if idx == 1 else f"c_{uuid.uuid4().hex[:8]}"
				sub.id = newId
				updated[newId] = sub

		self.concepts = updated
		self._refreshConceptTextsFromIndices()

	def _splitConceptSemantically(self, concept: Concept, maxSize: int) -> List[Concept]:
		local = []
		for localIdx, globalIdx in enumerate(concept.propositionIndices):
			if 0 <= globalIdx < len(self.allPropositions):
				item = self.allPropositions[globalIdx]
				text = item["text"] if isinstance(item, dict) else str(item)
				local.append((localIdx, globalIdx, text))

		if len(local) <= maxSize:
			return []

		lines = [f"{idx}. {text}" for idx, _global, text in local]
		userPrompt = (
			"Split the oversized educational concept into smaller, coherent teachable sub-concepts.\n"
			f"Constraints:\n"
			f"- Each sub-concept must contain between 4 and {maxSize} propositions when possible.\n"
			"- Keep propositions that teach the same learning objective together.\n"
			"- Use concise textbook-like titles.\n"
			"- Do NOT create a generic catch-all concept.\n"
			"- Sub-concept summaries must explicitly state a distinct learning objective.\n"
			"- Every proposition index must appear exactly once across subConcepts.\n\n"
			"Return JSON only:\n"
			'{\n'
			'  "subConcepts": [\n'
			'    {"title":"...", "summary":"...", "propositionIndices":[0,1,2]}\n'
			"  ]\n"
			"}\n\n"
			f"Oversized concept title: {concept.title}\n"
			f"Oversized concept summary: {concept.summary}\n\n"
			"Propositions (local indices):\n"
			f"{chr(10).join(lines)}\n"
		)

		try:
			raw = self.llm.chat(
				messages=[
					{"role": "system", "content": "You are splitting one educational concept into coherent sub-concepts."},
					{"role": "user", "content": userPrompt},
				],
				model=SEMANTIC_SPLIT_MODEL,
				response_format={"type": "json_object"},
			)
			data = json.loads(raw)
			subConceptsData = data.get("subConcepts", [])
			if not isinstance(subConceptsData, list) or not subConceptsData:
				return []
		except Exception as e:
			self.log(f"   Guard: semantic split failed for '{concept.title}': {e}")
			return []

		indexMap = {localIdx: globalIdx for localIdx, globalIdx, _text in local}
		usedLocal = set()
		result: List[Concept] = []

		for subData in subConceptsData:
			localIndices = subData.get("propositionIndices", [])
			if not isinstance(localIndices, list):
				continue
			validLocal = []
			for idx in localIndices:
				try:
					idxInt = int(idx)
				except Exception:
					continue
				if idxInt in indexMap and idxInt not in usedLocal:
					validLocal.append(idxInt)
					usedLocal.add(idxInt)

			if not validLocal:
				continue

			globalIndices = [indexMap[idx] for idx in validLocal]
			title = str(subData.get("title", "")).strip() or concept.title
			summary = str(subData.get("summary", "")).strip() or concept.summary
			result.append(Concept(
				id=f"c_{uuid.uuid4().hex[:8]}",
				title=title,
				summary=summary,
				propositionIndices=sorted(globalIndices),
				propositions=[],
			))

		# Preserve any dropped indices instead of losing assignment.
		missingLocal = [idx for idx in indexMap.keys() if idx not in usedLocal]
		if missingLocal:
			result.append(Concept(
				id=f"c_{uuid.uuid4().hex[:8]}",
				title=f"{concept.title} (Remainder)",
				summary=concept.summary,
				propositionIndices=sorted(indexMap[idx] for idx in missingLocal),
				propositions=[],
			))

		if len(result) <= 1:
			return []

		return result

	def _runConceptPurityPass(self, minConceptSize: int = 6):
		targets = [
			(cid, c) for cid, c in self.concepts.items()
			if len(c.propositionIndices) >= minConceptSize
		]
		if not targets:
			return

		self.log(f"   Guard: running concept purity pass on {len(targets)} concepts.")

		futures = []
		for conceptId, concept in targets:
			futures.append(self.llm.submit(self._purityKeepIndicesForConcept, conceptId, concept))

		for future in futures:
			try:
				conceptId, keepLocal = future.result()
			except Exception as e:
				self.log(f"   Guard: purity check failed: {e}")
				continue
			if conceptId not in self.concepts or not keepLocal:
				continue

			concept = self.concepts[conceptId]
			oldGlobal = concept.propositionIndices
			keepSet = set(keepLocal)
			newGlobal = [g for localIdx, g in enumerate(oldGlobal) if localIdx in keepSet]
			if not newGlobal:
				continue
			if len(newGlobal) < len(oldGlobal):
				self.log(
					f"   Guard: purity trimmed concept '{concept.title}' "
					f"from {len(oldGlobal)} to {len(newGlobal)} propositions."
				)
				concept.propositionIndices = newGlobal

		self._refreshConceptTextsFromIndices()

	def _purityKeepIndicesForConcept(self, conceptId: str, concept: Concept):
		lines = [f"{i}. {text}" for i, text in enumerate(concept.propositions)]
		prompt = (
			"Given a concept objective and its proposition list, keep only propositions that directly fit the objective.\n"
			"Return strict JSON:\n"
			'{\n'
			'  "keepIndices": [0,1,2],\n'
			'  "dropIndices": [3]\n'
			"}\n\n"
			"Rules:\n"
			"- Keep propositions that are directly on-topic for the objective.\n"
			"- Drop propositions that are tangential, off-topic, or belong to another concept.\n"
			"- Keep at least 60% unless the concept is clearly mixed.\n\n"
			f"Concept title: {concept.title}\n"
			f"Concept summary: {concept.summary}\n\n"
			f"Propositions:\n{chr(10).join(lines)}\n"
		)
		raw = self.llm.chat(
			messages=[
				{"role": "system", "content": "You are a strict concept-quality reviewer."},
				{"role": "user", "content": prompt},
			],
			model=PURITY_MODEL,
			response_format={"type": "json_object"},
		)
		data = json.loads(raw)
		keep = data.get("keepIndices", []) if isinstance(data, dict) else []
		keepLocal = []
		for idx in keep:
			try:
				i = int(idx)
			except Exception:
				continue
			if 0 <= i < len(concept.propositions):
				keepLocal.append(i)
		if not keepLocal:
			keepLocal = list(range(len(concept.propositions)))
		return conceptId, sorted(set(keepLocal))

	def _reattachTinyConcepts(self, minSize: int):
		if minSize <= 1:
			return
		tinyIds = [
			cid for cid, concept in self.concepts.items()
			if 0 < len(concept.propositionIndices) < minSize
		]
		if not tinyIds:
			return

		for cid in tinyIds:
			if cid not in self.concepts:
				continue
			concept = self.concepts[cid]
			isRemainder = "(Remainder)" in concept.title or concept.title == "Recovered Propositions"
			if not isRemainder and len(self.concepts) <= 6:
				continue

			moved = 0
		for idx in list(concept.propositionIndices):
			bestId = self._bestConceptByTokenOverlap(idx, excludeConceptId=cid)
			if bestId is None:
				bestId = self._bestConceptByThemeKeywords(idx, excludeConceptId=cid)
			if bestId:
				self.concepts[bestId].propositionIndices.append(idx)
				moved += 1
			if moved > 0:
				self.log(
					f"   Guard: reattached {moved} propositions from tiny concept '{concept.title}'."
				)
				concept.propositionIndices = []

		self._dropEmptyConcepts()
		self._refreshConceptTextsFromIndices()

	def _reattachMissingPropositions(self, missing: List[int]) -> List[int]:
		if not missing:
			return []

		conceptIds = list(self.concepts.keys())
		if not conceptIds:
			return missing

		catalogBlocks = []
		for conceptId, concept in self.concepts.items():
			sample = concept.propositions[:3]
			sampleText = "\n".join(f"- {s}" for s in sample)
			catalogBlocks.append(
				f"ID: {conceptId}\n"
				f"Title: {concept.title}\n"
				f"Summary: {concept.summary}\n"
				f"Sample propositions:\n{sampleText}\n"
			)
		catalog = "\n---\n".join(catalogBlocks)

		missingLines = []
		for idx in missing:
			if 0 <= idx < len(self.allPropositions):
				item = self.allPropositions[idx]
				text = item["text"] if isinstance(item, dict) else str(item)
				missingLines.append(f"{idx}. {text}")
		if not missingLines:
			return missing

		prompt = (
			"Assign each missing proposition to the best existing concept.\n"
			"Rules:\n"
			"- Use only listed concept IDs.\n"
			"- If no concept fits, leave it unassigned.\n"
			"- Return strict JSON.\n\n"
			"Output format:\n"
			'{\n'
			'  "assignments": [{"index": 12, "conceptId": "c_abc123"}],\n'
			'  "unassigned": [45]\n'
			"}\n\n"
			f"Existing concepts:\n{catalog}\n\n"
			f"Missing propositions:\n{chr(10).join(missingLines)}\n"
		)

		assignments = []
		unassigned = set(missing)
		try:
			raw = self.llm.chat(
				messages=[
					{"role": "system", "content": "You assign missing propositions to existing educational concepts."},
					{"role": "user", "content": prompt},
				],
				model=REATTACH_MODEL,
				response_format={"type": "json_object"},
			)
			data = json.loads(raw)
			assignments = data.get("assignments", []) if isinstance(data, dict) else []
			if isinstance(data, dict) and isinstance(data.get("unassigned"), list):
				unassigned = {int(x) for x in data.get("unassigned", []) if isinstance(x, int) or str(x).isdigit()}
		except Exception as e:
			self.log(f"   Guard: missing-proposition reattach failed: {e}")
			assignments = []

		for item in assignments:
			if not isinstance(item, dict):
				continue
			try:
				idx = int(item.get("index"))
			except Exception:
				continue
			conceptId = str(item.get("conceptId", "")).strip()
			if idx not in missing or conceptId not in self.concepts:
				continue
			if idx in self.concepts[conceptId].propositionIndices:
				unassigned.discard(idx)
				continue
			self.concepts[conceptId].propositionIndices.append(idx)
			unassigned.discard(idx)

		# Deterministic fallback for any remaining indices.
		for idx in list(unassigned):
			bestId = self._bestConceptByTokenOverlap(idx)
			if bestId is None:
				bestId = self._bestConceptByThemeKeywords(idx)
			if bestId:
				self.concepts[bestId].propositionIndices.append(idx)
				unassigned.discard(idx)

		stillMissing = sorted(idx for idx in missing if idx in unassigned)
		if stillMissing:
			self.log(f"   Guard: could not reattach {len(stillMissing)} propositions; keeping recovery fallback.")
		else:
			self.log("   Guard: reattached all missing propositions to existing concepts.")
		return stillMissing

	def _bestConceptByTokenOverlap(self, propIndex: int, excludeConceptId: Optional[str] = None) -> Optional[str]:
		if not (0 <= propIndex < len(self.allPropositions)):
			return None
		item = self.allPropositions[propIndex]
		text = item["text"] if isinstance(item, dict) else str(item)
		propTokens = set(self._normalizePropText(text).split())
		if not propTokens:
			return None

		bestId = None
		bestScore = 0.0
		for conceptId, concept in self.concepts.items():
			if excludeConceptId and conceptId == excludeConceptId:
				continue
			if len(concept.propositionIndices) >= MAX_PROPS_PER_CONCEPT:
				continue
			conceptTokens = set()
			for t in concept.propositions[:10]:
				conceptTokens.update(self._normalizePropText(t).split())
			if not conceptTokens:
				continue
			score = len(propTokens & conceptTokens) / max(1, len(propTokens | conceptTokens))
			if score > bestScore:
				bestScore = score
				bestId = conceptId
		if bestScore < 0.08:
			return None
		return bestId

	def _bestConceptByThemeKeywords(self, propIndex: int, excludeConceptId: Optional[str] = None) -> Optional[str]:
		if not (0 <= propIndex < len(self.allPropositions)):
			return None
		item = self.allPropositions[propIndex]
		text = item["text"] if isinstance(item, dict) else str(item)
		textTokens = set(self._normalizePropText(text).split())
		if not textTokens:
			return None

		themeScores = {
			theme: len(textTokens & keywords)
			for theme, keywords in THEME_KEYWORDS.items()
		}
		bestTheme = max(themeScores, key=themeScores.get)
		if themeScores[bestTheme] == 0:
			return None

		bestId = None
		bestScore = 0
		keywords = THEME_KEYWORDS[bestTheme]
		for conceptId, concept in self.concepts.items():
			if excludeConceptId and conceptId == excludeConceptId:
				continue
			if len(concept.propositionIndices) >= MAX_PROPS_PER_CONCEPT:
				continue
			signature = self._normalizePropText(f"{concept.title} {concept.summary}")
			sigTokens = set(signature.split())
			score = len(sigTokens & keywords)
			if score > bestScore:
				bestScore = score
				bestId = conceptId
		if bestScore == 0:
			return None
		return bestId

	def _reduceInterConceptOverlap(self, threshold: float):
		if threshold <= 0:
			return
		ids = list(self.concepts.keys())
		for i in range(len(ids)):
			idA = ids[i]
			if idA not in self.concepts:
				continue
			for j in range(i + 1, len(ids)):
				idB = ids[j]
				if idB not in self.concepts:
					continue
				aTexts = {self._normalizePropText(t) for t in self.concepts[idA].propositions}
				bTexts = {self._normalizePropText(t) for t in self.concepts[idB].propositions}
				aTexts.discard("")
				bTexts.discard("")
				a = set(self.concepts[idA].propositionIndices)
				b = set(self.concepts[idB].propositionIndices)
				if not a or not b:
					continue
				inter = aTexts & bTexts
				if not inter:
					continue
				overlapRatio = len(inter) / min(len(aTexts), len(bTexts))
				if overlapRatio <= threshold:
					continue

				keepId = idA
				dropId = idB
				if len(b) > len(a) or (len(b) == len(a) and idB < idA):
					keepId = idB
					dropId = idA
				self.log(
					f"   Guard: overlap {overlapRatio:.2f} between '{idA}' and '{idB}', "
					f"keeping shared propositions in '{keepId}'."
				)
				filtered = []
				for idx in self.concepts[dropId].propositionIndices:
					if 0 <= idx < len(self.allPropositions):
						item = self.allPropositions[idx]
						text = item["text"] if isinstance(item, dict) else str(item)
						if self._normalizePropText(text) in inter:
							continue
					filtered.append(idx)
				self.concepts[dropId].propositionIndices = sorted(set(filtered))

		self._refreshConceptTextsFromIndices()

	def _dropEmptyConcepts(self):
		emptyIds = [cid for cid, c in self.concepts.items() if not c.propositionIndices]
		for cid in emptyIds:
			del self.concepts[cid]
		if emptyIds:
			self.log(f"   Guard: dropped {len(emptyIds)} empty concepts.")

	def _validateAssignments(self):
		total = len(self.allPropositions)
		owners = {}
		for cid, concept in self.concepts.items():
			for idx in concept.propositionIndices:
				owners.setdefault(idx, []).append(cid)

		missing = [idx for idx in range(total) if idx not in owners]
		duplicates = [idx for idx, cids in owners.items() if len(cids) > 1]
		if missing or duplicates:
			raise ValueError(
				f"Assignment validation failed: missing={len(missing)}, duplicate_owners={len(duplicates)}"
			)
