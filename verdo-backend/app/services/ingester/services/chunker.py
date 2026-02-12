import json
import re
from typing import Any, Dict, List

import numpy as np
import tiktoken

from app.services.ingester.prompts import DECOMPOSE_PROPOSITIONS_PROMPT
from app.services.ingester.services.Embedder import Embedder
from app.services.ingester.services.HDBSCANplus import HDBSCANplus
from app.services.ingester.services.LLM import LLM
from app.services.ingester.services.chunk import Chunk

FIGURE_REF_RE = re.compile(r"\[FIGURE ([^\]]+)\]")
UNRESOLVED_START_RE = re.compile(r"^\s*(this|that|it|these|those|they|such)\b", re.IGNORECASE)
RELATIVE_REF_RE = re.compile(
	r"\b(as (shown|noted|mentioned) (above|below)|see (above|below)|the (figure|table) above)\b",
	re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
META_CONTENT_RE = re.compile(
	r"^\s*(this|the)\s+(chapter|section)\b|^\s*in\s+this\s+(chapter|section)\b",
	re.IGNORECASE,
)

# --- Classes ------------------------------------------------------------

class Chunker:
	def __init__(self):
		self.elements = []           # List of content strings (non-figure)
		self.elementMetas = []       # List of dicts {id, kind} matching self.elements
		self.figuresById = {}        # Map figure_id -> figure_json
		self.order = []              # List of ALL element IDs in reading order
		self.idToElementIndex = {}   # Map non-figure ID -> index in self.elements
		
		self.propositions = []       # List[dict] with {text, batchIndex, sourceElementIds}
		self.batches = []            # list of {indices: [], text: ""}
		self.chunks = []
		
		# Semantic Clustering State
		self.propositionEmbeddings = []   # List[List[float]]
		self.clusterResult = None         # HDBSCANplus result
		self.semanticChunks = []          # Final output of this stage

	# Extracts content from elements in the provided JSON data
	# Handles 'table' types specifically
	def getElements(self, jsonData: List[Dict[str, Any]]):
		self.elements = []
		self.elementMetas = []
		self.figuresById = {}
		self.order = []
		self.idToElementIndex = {}

		for page in jsonData:
			for element in page.get('elements', []):
				el_id = element.get('id')
				if not el_id:
					continue

				kind = element.get('type')
				content = ""
				
				# Always track global order
				self.order.append(el_id)

				# Handle Figures/Images/Math separately - do not add to decompose list
				if kind in ['image', 'figure', 'math']:
					self.figuresById[el_id] = element
					continue

				# Extract text content for others
				if kind == 'table':
					contentObj = element.get('content')
					if isinstance(contentObj, dict):
						content = contentObj.get('markdown', "")
					else:
						content = str(contentObj)
				else:
					content = element.get('content', "")

				contentStr = str(content).strip()
				if contentStr:
					idx = len(self.elements)
					self.elements.append(contentStr)
					self.elementMetas.append({"id": el_id, "kind": kind})
					self.idToElementIndex[el_id] = idx

	# Batches elements based on a token threshold using tiktoken
	def batch(self, threshold: int = 500):
		encoder = tiktoken.get_encoding("cl100k_base")

		self.batches = []
		currentIndices = []
		currentTextToks = 0

		for idx, element in enumerate(self.elements):
			elementTokens = len(encoder.encode(element))

			if currentTextToks + elementTokens < threshold:
				currentIndices.append(idx)
				currentTextToks += elementTokens
			else:
				if currentIndices:
					# Store batch info
					batchText = "\n".join([self.elements[i] for i in currentIndices])
					self.batches.append({
						"indices": currentIndices,
						"text": batchText
					})
				
				currentIndices = [idx]
				currentTextToks = elementTokens

		if currentIndices:
			batchText = "\n".join([self.elements[i] for i in currentIndices])
			self.batches.append({
				"indices": currentIndices,
				"text": batchText
			})

	# Build context window including surrounding elements and figures
	def _buildContextWindow(self, batchIndices: List[int], windowRadius: int = 3):
		if not batchIndices:
			return []

		# 1. Collect IDs of elements in this batch
		batchIds = [self.elementMetas[i]["id"] for i in batchIndices]

		# 2. Find min and max positions in global order for these IDs
		#    Build a mapping from id -> position in self.order for efficiency
		idToOrderPos = {elId: pos for pos, elId in enumerate(self.order)}

		batchPositions = [idToOrderPos[elId] for elId in batchIds if elId in idToOrderPos]
		if not batchPositions:
			return []

		minPos = min(batchPositions)
		maxPos = max(batchPositions)

		# 3. Expand the window by windowRadius on both sides
		start = max(0, minPos - windowRadius)
		end = min(len(self.order) - 1, maxPos + windowRadius)

		# 4. Build context strings from this slice of self.order
		contextItems = []
		for pos in range(start, end + 1):
			elId = self.order[pos]

			# Case A: non-figure element (has text)
			if elId in self.idToElementIndex:
				textIndex = self.idToElementIndex[elId]
				text = self.elements[textIndex]
				contextItems.append(text)
				continue

			# Case B: figure element (no direct text – keep placeholder)
			if elId in self.figuresById:
				contextItems.append(f"[FIGURE {elId}]")
				continue

		return contextItems

	# Generates propositions from batches using LLM
	def getPropositions(self):
		self.llm = LLM()
		self.propositions = []
		seenNormalized = set()
		validFigureIds = set(self.figuresById.keys())

		# Prepare futures for all batches
		futures = []
		for i, batchData in enumerate(self.batches):
			batchText = batchData["text"]
			batchIndices = batchData["indices"]
			
			# Build Context
			contextItems = self._buildContextWindow(batchIndices, windowRadius=5)
			contextText = "\n\n".join(contextItems)
			
			prompt = DECOMPOSE_PROPOSITIONS_PROMPT.format(context=contextText, batch=batchText)
			
			# Using chatAsync which returns a Future
			futures.append(self.llm.chatAsync(
				messages=[{"role": "user", "content": prompt}], 
				model="gpt-5-nano", 
				timeout=120,
				response_format={"type": "json_object"}
			))

		# Wait for all futures to complete
		for i, future in enumerate(futures):
			try:
				response = future.result()
				try:
					data = json.loads(response)
					props = data.get("propositions", [])
					sourceIds = [self.elementMetas[idx]["id"] for idx in self.batches[i]["indices"]]
					batchIndex = i + 1  # 1-indexed for human readability
					
					if isinstance(props, list):
						for propText in props:
							cleaned = self._cleanPropositionText(str(propText), validFigureIds)
							if not cleaned:
								continue
							normalized = self._normalizeForDedup(cleaned)
							if normalized in seenNormalized:
								continue
							seenNormalized.add(normalized)
							self.propositions.append({
								"text": cleaned,
								"batchIndex": batchIndex,
								"sourceElementIds": sourceIds
							})
					else:
						# Fallback if top level is list or other issue
						if isinstance(data, list):
							for propText in data:
								cleaned = self._cleanPropositionText(str(propText), validFigureIds)
								if not cleaned:
									continue
								normalized = self._normalizeForDedup(cleaned)
								if normalized in seenNormalized:
									continue
								seenNormalized.add(normalized)
								self.propositions.append({
									"text": cleaned,
									"batchIndex": batchIndex,
									"sourceElementIds": sourceIds
								})
						else:
							print(f"Warning: JSON output in batch {i} did not contain 'propositions' list.")
				except json.JSONDecodeError as e:
					print(f"JSON Decode Error in batch {i}: {e}")
					print(f"Response snippet: {response[:100]}...")
			except Exception as e:
				print(f"Error processing batch {i}: {e}")

	def _cleanPropositionText(self, text: str, validFigureIds: set) -> str:
		cleaned = " ".join(text.replace("\n", " ").split()).strip()
		if not cleaned:
			return ""

		# Drop unresolved context-dependent starts.
		if UNRESOLVED_START_RE.search(cleaned):
			return ""

		# Drop relative references that require layout context.
		if RELATIVE_REF_RE.search(cleaned):
			return ""

		# Keep only known figure references.
		def _figureRepl(match):
			figId = match.group(1)
			return match.group(0) if figId in validFigureIds else ""

		cleaned = FIGURE_REF_RE.sub(_figureRepl, cleaned)
		cleaned = " ".join(cleaned.split()).strip()
		if not cleaned:
			return ""

		# Drop chapter/section boilerplate statements.
		if META_CONTENT_RE.search(cleaned):
			return ""

		# Reject trivial statements.
		tokenCount = len(TOKEN_RE.findall(cleaned))
		if tokenCount < 5:
			return ""

		return cleaned

	def _normalizeForDedup(self, text: str) -> str:
		withoutFigures = FIGURE_REF_RE.sub("", text)
		return re.sub(r"\s+", " ", withoutFigures.lower()).strip().strip(".")

	# Embeds all extracted propositions
	def embedPropositions(self):
		embedder = Embedder()
		self.propositionEmbeddings = []
		
		for prop in self.propositions:
			propText = prop["text"] if isinstance(prop, dict) else prop
			embedding = embedder.getEmbedding(propText)
			self.propositionEmbeddings.append(embedding)
			
		if len(self.propositionEmbeddings) != len(self.propositions):
			raise ValueError("Mismatch between proposition count and embedding count.")
			
		if not self.propositionEmbeddings:
			print("Warning: No embeddings generated (empty propositions list?).")

	# Clusters stored embeddings using HDBSCANplus
	def clusterPropositions(self):
		if not self.propositionEmbeddings:
			print("No embeddings to cluster.")
			return

		# Convert embeddings to numpy array
		X = np.array(self.propositionEmbeddings, dtype=np.float32)
		

		# Instantiate HDBSCANplus - self-converging parameter search
		hdb = HDBSCANplus(
			metric="cosine",
			normalizeVectors=True,
			maxTrials=120,
		)
		
		# Run fitPredict
		result = hdb.fitPredict(X)
		self.clusterResult = result

	# Builds semantic chunks from clustering results
	def buildSemanticChunks(self):
		if not self.clusterResult:
			print("No clustering results available.")
			return
			
		labels = self.clusterResult.labels
		probabilities = self.clusterResult.probabilities
		
		# Organize indices by label
		clusters = {}
		for idx, label in enumerate(labels):
			if label == -1:
				continue # Skip noise for now
			
			if label not in clusters:
				clusters[label] = []
			clusters[label].append(idx)
			
		embedder = Embedder() # Used for intra-cluster dedup
		self.semanticChunks = []
		
		for label in sorted(clusters.keys()):
			indices = clusters[label]
			chunkPropositions = []
			acceptedIndices = []
			figureIds = set()
			sourceElementIds = set()
			
			# Calculate confidence for the cluster
			clusterProbs = [probabilities[i] for i in indices]
			confidence = np.mean(clusterProbs) if clusterProbs else 0.0
			
			for idx in indices:
				propText = self.propositions[idx]
				propEmbedding = self.propositionEmbeddings[idx]
				
				# Intra-cluster Deduplication
				isDuplicate = False
				for existingIdx in acceptedIndices:
					existingEmbedding = self.propositionEmbeddings[existingIdx]
					sim = embedder.cosineSimilarity(propEmbedding, existingEmbedding)
					if sim >= 0.985:
						isDuplicate = True
						break
				
				if not isDuplicate:
					chunkPropositions.append(propText)
					acceptedIndices.append(idx)
					sourceElementIds.update(self.propositionSources[idx])
					figureIds.update(re.findall(r"\[FIGURE ([^\]]+)\]", propText))

			chunkEmbedding = None
			if acceptedIndices:
				acceptedEmbeddings = [self.propositionEmbeddings[i] for i in acceptedIndices]
				chunkEmbedding = np.mean(np.array(acceptedEmbeddings, dtype=np.float32), axis=0).tolist()

			# Create chunk object
			chunk = Chunk(
				f"chunk_{len(self.semanticChunks):04d}",
				chunkPropositions,
				confidence=confidence,
				embedding=chunkEmbedding,
				figure_ids=sorted(figureIds),
				source_element_ids=sorted(sourceElementIds),
				relations={},
			)
			self.semanticChunks.append(chunk)
