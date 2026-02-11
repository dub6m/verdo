import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.services.ingester.prompts import (
	CONCEPT_CREATION_SYSTEM_PROMPT,
	CONCEPT_CREATION_USER_PROMPT,
	CONCEPT_MERGE_SYSTEM_PROMPT,
	CONCEPT_MERGE_USER_PROMPT,
)
from app.services.ingester.services.LLM import LLM

# --- Constants --------------------------------------------------------------

PHASE1_BATCH_SIZE = 40  # propositions per batch (increased for better context)
PHASE2_MERGE_BATCH_SIZE = 15  # concepts per merge batch
PHASE2_OVERLAP_SIZE = 4  # concepts from each side of boundary

# --- Data Classes -----------------------------------------------------------

@dataclass
class Concept:
	id: str
	title: str
	summary: str
	propositions: List[str] = field(default_factory=list)
	propositionIndices: List[int] = field(default_factory=list)  # Global indices
	originalBatchIndex: int = 0  # For tracking merge boundaries
	prerequisites: List[str] = field(default_factory=list)  # For Phase 3

	def toDict(self) -> dict:
		return asdict(self)


# --- Main Class -------------------------------------------------------------

class ConceptBuilder:
	"""
	Builds educational concepts from propositions using LLM-based grouping.
	
	Phase 1: Group propositions into concepts within batches
	Phase 2: Merge concepts across batch boundaries (2 rounds)
	Phase 3: Extract prerequisites (TBD)
	"""

	def __init__(self, llmClient=None, printLogging: bool = True):
		self.llm = llmClient or LLM(maxWorkers=30)
		self.printLogging = printLogging
		self.concepts: Dict[str, Concept] = {}
		self.allPropositions: List[dict] = []
		self.skippedIndices: List[int] = []  # Track skipped meta-content

	def buildConcepts(self, propositions: List[dict]) -> Dict[str, Concept]:
		"""
		Main entry point. Takes propositions and returns concepts.
		
		Args:
			propositions: List of dicts with {text, batchIndex, sourceElementIds}
		
		Returns:
			Dict of concept_id -> Concept
		"""
		self.allPropositions = propositions

		if self.printLogging:
			print(f"\n{'='*70}")
			print(f"🧠 CONCEPT BUILDER")
			print(f"{'='*70}")
			print(f"Total propositions: {len(propositions)}")

		# Phase 1: Create concepts from proposition batches
		self._phase1CreateConcepts()
		conceptsBeforeMerge = len(self.concepts)

		# Phase 2: Merge concepts (2 rounds)
		self._phase2MergeConcepts()
		conceptsAfterMerge = len(self.concepts)

		if self.printLogging:
			print(f"\n{'='*70}")
			print(f"✅ CONCEPT BUILDING COMPLETE")
			print(f"{'='*70}")
			print(f"Concepts before merge: {conceptsBeforeMerge}")
			print(f"Concepts after merge: {conceptsAfterMerge}")
			print(f"Skipped propositions: {len(self.skippedIndices)}")
			
			# Show distribution of concept sizes
			sizes = [len(c.propositions) for c in self.concepts.values()]
			if sizes:
				avg = sum(sizes) / len(sizes)
				print(f"Avg propositions per concept: {avg:.1f}")
				print(f"Distribution: min={min(sizes)}, max={max(sizes)}")
				
				# Count concepts by size
				single = sum(1 for s in sizes if s == 1)
				small = sum(1 for s in sizes if 2 <= s <= 3)
				medium = sum(1 for s in sizes if 4 <= s <= 7)
				large = sum(1 for s in sizes if s >= 8)
				print(f"Size breakdown: 1-prop={single}, 2-3={small}, 4-7={medium}, 8+={large}")

		return self.concepts

	# -------------------------------------------------------------------------
	# Phase 1: Create Concepts
	# -------------------------------------------------------------------------

	def _phase1CreateConcepts(self):
		"""Batch propositions and create concepts via LLM."""
		if self.printLogging:
			print(f"\n📦 PHASE 1: Creating concepts from propositions")

		# Create batches of propositions
		batches = []
		for i in range(0, len(self.allPropositions), PHASE1_BATCH_SIZE):
			batch = self.allPropositions[i:i + PHASE1_BATCH_SIZE]
			batches.append({
				"startIndex": i,
				"propositions": batch
			})

		if self.printLogging:
			print(f"   Batches: {len(batches)} (batch size: {PHASE1_BATCH_SIZE})")

		# Process batches in parallel
		futures = []
		for batchIdx, batchData in enumerate(batches):
			futures.append(self.llm.submit(
				self._processBatchPhase1,
				batchData["propositions"],
				batchData["startIndex"],
				batchIdx
			))

		# Collect results
		for future in futures:
			try:
				batchConcepts, batchSkipped = future.result()
				for concept in batchConcepts:
					self.concepts[concept.id] = concept
				self.skippedIndices.extend(batchSkipped)
			except Exception as e:
				print(f"   ⚠️ Error in Phase 1 batch: {e}")

		if self.printLogging:
			print(f"   Created {len(self.concepts)} initial concepts")
			print(f"   Skipped {len(self.skippedIndices)} meta-content propositions")

	def _processBatchPhase1(
		self,
		batchPropositions: List[dict],
		globalStartIndex: int,
		batchIdx: int
	) -> tuple:
		"""Process a single batch of propositions to create concepts.
		
		Returns:
			Tuple of (concepts list, skipped indices list)
		"""
		# Format propositions for the prompt
		propLines = []
		for i, prop in enumerate(batchPropositions):
			propText = prop["text"] if isinstance(prop, dict) else prop
			propLines.append(f"{i}. {propText}")
		
		propositionsText = "\n".join(propLines)
		
		userPrompt = CONCEPT_CREATION_USER_PROMPT.format(
			count=len(batchPropositions),
			max_index=len(batchPropositions) - 1,
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
			skippedLocal = data.get("skipped", [])

			# Convert skipped to global indices
			skippedGlobal = [globalStartIndex + i for i in skippedLocal]

			# Convert to Concept objects
			concepts = []
			for cData in conceptsData:
				localIndices = cData.get("propositionIndices", [])
				globalIndices = [globalStartIndex + i for i in localIndices]
				
				# Get proposition texts
				propTexts = []
				for i in localIndices:
					if 0 <= i < len(batchPropositions):
						prop = batchPropositions[i]
						propTexts.append(prop["text"] if isinstance(prop, dict) else prop)

				# Generate a proper UUID for this concept
				conceptId = f"c_{uuid.uuid4().hex[:8]}"
				
				concept = Concept(
					id=conceptId,
					title=cData.get("title", "Untitled"),
					summary=cData.get("summary", ""),
					propositions=propTexts,
					propositionIndices=globalIndices,
					originalBatchIndex=batchIdx
				)
				concepts.append(concept)

			return concepts, skippedGlobal

		except Exception as e:
			print(f"   ⚠️ Error parsing Phase 1 response: {e}")
			# Fallback: each proposition becomes its own concept
			fallbackConcepts = []
			for i, prop in enumerate(batchPropositions):
				propText = prop["text"] if isinstance(prop, dict) else prop
				concept = Concept(
					id=f"c_{uuid.uuid4().hex[:8]}",
					title=f"Concept from proposition {globalStartIndex + i}",
					summary=propText[:100] + "..." if len(propText) > 100 else propText,
					propositions=[propText],
					propositionIndices=[globalStartIndex + i],
					originalBatchIndex=batchIdx
				)
				fallbackConcepts.append(concept)
			return fallbackConcepts, []

	# -------------------------------------------------------------------------
	# Phase 2: Merge Concepts
	# -------------------------------------------------------------------------

	def _phase2MergeConcepts(self):
		"""Merge concepts in 3 rounds: standard, overlapping, then small concept cleanup."""
		if self.printLogging:
			print(f"\n🔄 PHASE 2: Merging concepts")
			print(f"   Starting with {len(self.concepts)} concepts")

		# Round 1: Standard batches
		countBefore = len(self.concepts)
		if self.printLogging:
			print(f"   Round 1: Standard merge batches")
		self._mergeRound(useOverlap=False)
		if self.printLogging:
			print(f"   After Round 1: {len(self.concepts)} concepts (merged {countBefore - len(self.concepts)})")

		# Round 2: Overlapping batches to bridge boundaries
		countBefore = len(self.concepts)
		if self.printLogging:
			print(f"   Round 2: Overlapping merge batches")
		self._mergeRound(useOverlap=True)
		if self.printLogging:
			print(f"   After Round 2: {len(self.concepts)} concepts (merged {countBefore - len(self.concepts)})")

		# Round 3: Small concept cleanup - merge single/small concepts into neighbors
		countBefore = len(self.concepts)
		if self.printLogging:
			print(f"   Round 3: Small concept cleanup")
		self._mergeSmallConcepts()
		if self.printLogging:
			print(f"   After Round 3: {len(self.concepts)} concepts (merged {countBefore - len(self.concepts)})")

	def _mergeSmallConcepts(self):
		"""Find single-proposition concepts and try to merge them with related larger concepts."""
		# Get all concepts sorted by document order (using first proposition index)
		sortedConcepts = sorted(
			self.concepts.values(),
			key=lambda c: min(c.propositionIndices) if c.propositionIndices else 0
		)

		# Find small concepts (1-2 propositions)
		smallConcepts = [c for c in sortedConcepts if len(c.propositions) <= 2]
		largerConcepts = [c for c in sortedConcepts if len(c.propositions) > 2]

		if not smallConcepts or not largerConcepts:
			return

		if self.printLogging:
			print(f"      Found {len(smallConcepts)} small concepts to evaluate")

		# Process in batches: each small concept paired with its neighbors
		for smallConcept in smallConcepts[:]:  # Copy list since we modify during iteration
			if smallConcept.id not in self.concepts:
				continue  # Already merged

			# Find nearby larger concepts (by document position)
			smallPos = min(smallConcept.propositionIndices) if smallConcept.propositionIndices else 0
			
			# Get closest larger concepts
			neighbors = sorted(
				[c for c in largerConcepts if c.id in self.concepts],
				key=lambda c: abs(min(c.propositionIndices) - smallPos) if c.propositionIndices else float('inf')
			)[:3]  # Top 3 closest neighbors

			if not neighbors:
				continue

			# Ask LLM if this small concept should merge with any neighbor
			self._tryMergeSmallWithNeighbors(smallConcept, neighbors)

	def _tryMergeSmallWithNeighbors(self, smallConcept: Concept, neighbors: List[Concept]):
		"""Ask LLM if a small concept should be absorbed into one of its neighbors."""
		# Build prompt
		smallDesc = (
			f"SMALL CONCEPT:\n"
			f"  Title: {smallConcept.title}\n"
			f"  Propositions: {smallConcept.propositions}\n"
		)

		neighborDescs = []
		for i, n in enumerate(neighbors):
			propSample = n.propositions[:3]
			neighborDescs.append(
				f"{i+1}. \"{n.title}\" ({len(n.propositions)} propositions)\n"
				f"   Sample: {propSample[:2]}..."
			)
		neighborsText = "\n".join(neighborDescs)

		prompt = f"""Should this small concept be merged into one of the larger concepts below?

{smallDesc}

CANDIDATE CONCEPTS TO MERGE INTO:
{neighborsText}

RULES:
- Only merge if the small concept clearly belongs as part of the larger one
- If it's a distinct topic, keep it separate (return "none")
- Meta-content (author names, chapter intros) should be marked for deletion

Return JSON: {{"action": "merge", "targetIndex": 1}} or {{"action": "keep"}} or {{"action": "delete"}}
"""

		try:
			response = self.llm.chat(
				messages=[{"role": "user", "content": prompt}],
				model="gpt-5-nano",
				response_format={"type": "json_object"}
			)

			data = json.loads(response)
			action = data.get("action", "keep")

			if action == "delete":
				# Remove meta-content
				del self.concepts[smallConcept.id]
				if self.printLogging:
					print(f"         🗑️ Deleted meta-content: \"{smallConcept.title}\"")
				return

			if action == "merge":
				targetIdx = data.get("targetIndex", 1) - 1  # Convert to 0-indexed
				if 0 <= targetIdx < len(neighbors):
					target = neighbors[targetIdx]
					if target.id in self.concepts and smallConcept.id in self.concepts:
						# Merge into target
						target.propositions.extend(smallConcept.propositions)
						target.propositionIndices.extend(smallConcept.propositionIndices)
						del self.concepts[smallConcept.id]
						if self.printLogging:
							print(f"         ↪️ Absorbed \"{smallConcept.title}\" into \"{target.title}\"")

		except Exception as e:
			pass  # Keep concept as-is on error

	def _mergeRound(self, useOverlap: bool):
		"""Execute one round of merging."""
		# Sort concepts by their original batch index for consistent ordering
		sortedConcepts = sorted(
			self.concepts.values(),
			key=lambda c: (c.originalBatchIndex, c.id)
		)

		if len(sortedConcepts) <= 1:
			return

		# Create batches for merging
		if useOverlap:
			# Overlapping batches: bridge the gaps between original batches
			batches = self._createOverlapBatches(sortedConcepts)
		else:
			# Standard batches
			batches = []
			for i in range(0, len(sortedConcepts), PHASE2_MERGE_BATCH_SIZE):
				batches.append(sortedConcepts[i:i + PHASE2_MERGE_BATCH_SIZE])

		if not batches:
			return

		if self.printLogging:
			print(f"      Processing {len(batches)} merge batches")

		# Process merge batches (sequential to avoid conflicts)
		for batchNum, batch in enumerate(batches):
			if len(batch) < 2:
				continue
			self._processMergeBatch(batch, batchNum)

	def _createOverlapBatches(self, sortedConcepts: List[Concept]) -> List[List[Concept]]:
		"""Create overlapping batches that bridge original batch boundaries."""
		batches = []
		
		# Group concepts by their original batch index
		batchGroups: Dict[int, List[Concept]] = {}
		for c in sortedConcepts:
			if c.originalBatchIndex not in batchGroups:
				batchGroups[c.originalBatchIndex] = []
			batchGroups[c.originalBatchIndex].append(c)

		# Create bridging batches between adjacent original batches
		sortedBatchIndices = sorted(batchGroups.keys())
		for i in range(len(sortedBatchIndices) - 1):
			currentIdx = sortedBatchIndices[i]
			nextIdx = sortedBatchIndices[i + 1]

			currentConcepts = batchGroups[currentIdx]
			nextConcepts = batchGroups[nextIdx]

			# Take last N from current, first N from next
			overlapFromCurrent = currentConcepts[-PHASE2_OVERLAP_SIZE:]
			overlapFromNext = nextConcepts[:PHASE2_OVERLAP_SIZE]

			bridgeBatch = overlapFromCurrent + overlapFromNext
			if len(bridgeBatch) >= 2:
				batches.append(bridgeBatch)

		return batches

	def _processMergeBatch(self, concepts: List[Concept], batchNum: int):
		"""Process a batch of concepts to identify merges."""
		# Format concepts for the prompt - include the actual proposition texts for context
		conceptLines = []
		conceptIds = []
		for c in concepts:
			conceptIds.append(c.id)
			# Show first 2 propositions as sample
			propSample = c.propositions[:2]
			propPreview = "\n".join([f"     - {p[:80]}..." if len(p) > 80 else f"     - {p}" for p in propSample])
			if len(c.propositions) > 2:
				propPreview += f"\n     ... and {len(c.propositions) - 2} more"
			
			conceptLines.append(
				f"ID: {c.id}\n"
				f"Title: {c.title}\n"
				f"Summary: {c.summary}\n"
				f"Propositions ({len(c.propositions)}):\n{propPreview}\n"
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

			if self.printLogging and mergeGroups:
				print(f"      Batch {batchNum}: {len(mergeGroups)} merge groups found")

			# Apply merges
			for group in mergeGroups:
				idsToMerge = group.get("conceptIds", [])
				if len(idsToMerge) < 2:
					continue

				# Verify all IDs exist in current concepts (re-check since previous merges may have deleted some)
				validIds = [cid for cid in idsToMerge if cid in self.concepts]
				if len(validIds) < 2:
					if self.printLogging:
						print(f"         ⚠️ Skipping merge - only {len(validIds)} valid IDs found of {len(idsToMerge)} requested")
					continue

				# Merge into the first concept
				primaryId = validIds[0]
				if primaryId not in self.concepts:
					continue  # Safety check
				primaryConcept = self.concepts[primaryId]

				# Update title and summary
				primaryConcept.title = group.get("mergedTitle", primaryConcept.title)
				primaryConcept.summary = group.get("mergedSummary", primaryConcept.summary)

				# Merge propositions from other concepts
				mergedCount = 1
				for otherId in validIds[1:]:
					if otherId not in self.concepts:
						continue  # Already merged in a previous group
					otherConcept = self.concepts[otherId]
					primaryConcept.propositions.extend(otherConcept.propositions)
					primaryConcept.propositionIndices.extend(otherConcept.propositionIndices)
					# Remove the merged concept
					del self.concepts[otherId]
					mergedCount += 1

				if self.printLogging and mergedCount > 1:
					print(f"         ✓ Merged {mergedCount} concepts -> \"{primaryConcept.title}\" ({len(primaryConcept.propositions)} props)")

		except Exception as e:
			print(f"   ⚠️ Error in merge batch {batchNum}: {e}")
			import traceback
			traceback.print_exc()

	# -------------------------------------------------------------------------
	# Output Methods
	# -------------------------------------------------------------------------

	def getConceptsList(self) -> List[dict]:
		"""Return concepts as a list of dictionaries."""
		return [c.toDict() for c in self.concepts.values()]

	def getConceptById(self, conceptId: str) -> Optional[dict]:
		"""Get a specific concept by ID."""
		if conceptId in self.concepts:
			return self.concepts[conceptId].toDict()
		return None

	def getStats(self) -> dict:
		"""Get statistics about the concept building process."""
		if not self.concepts:
			return {
				"totalConcepts": 0,
				"totalPropositions": len(self.allPropositions),
				"skippedPropositions": len(self.skippedIndices),
				"avgPropositionsPerConcept": 0,
			}

		propCounts = [len(c.propositions) for c in self.concepts.values()]
		return {
			"totalConcepts": len(self.concepts),
			"totalPropositions": len(self.allPropositions),
			"skippedPropositions": len(self.skippedIndices),
			"avgPropositionsPerConcept": sum(propCounts) / len(propCounts),
			"minPropositionsPerConcept": min(propCounts),
			"maxPropositionsPerConcept": max(propCounts),
			"singlePropConcepts": sum(1 for s in propCounts if s == 1),
		}
