import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from app.services.ingester.services.LLM import LLM
from app.services.ingester.services.chunk import Chunk

ANN_PLANES = 12
ANN_SEED = 42
ANN_TABLES = 6


@dataclass
class GraphNode:
	id: str
	type: str
	data: Optional[dict] = None
	metadata: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
	from_id: str
	to_id: str
	type: str
	evidence: str
	description: str
	kind: Optional[str] = None
	score: Optional[float] = None

	def edgeId(self) -> str:
		return f"{self.from_id}::{self.type}::{self.to_id}"


class IngestionGraph:
	def __init__(self):
		self.nodes: Dict[str, GraphNode] = {}
		self.edges: List[GraphEdge] = []
		self._edgeIds = set()

	def addNode(self, node: GraphNode):
		self.nodes[node.id] = node

	def addEdge(self, edge: GraphEdge):
		edgeId = edge.edgeId()
		if edgeId in self._edgeIds:
			return
		self._edgeIds.add(edgeId)
		self.edges.append(edge)


def buildGraph(
	chunks: List[Chunk],
	figuresById: Dict[str, dict],
	relatedTopK: int = 3,
	relatedThreshold: float = 0.8,
	dependencyTopK: int = 3,
	dependencyThreshold: float = 0.85,
	enableDependencyEdges: bool = True,
	llm: Optional[LLM] = None,
) -> IngestionGraph:
	graph = IngestionGraph()

	for chunk in chunks:
		graph.addNode(GraphNode(
			id=chunk.id,
			type="chunk",
			data={"propositions": chunk.data},
			metadata={
				"confidence": chunk.confidence,
				"embedding": chunk.embedding,
				"figure_ids": chunk.figure_ids,
				"source_element_ids": chunk.source_element_ids,
			},
		))

	for figureId, payload in figuresById.items():
		graph.addNode(GraphNode(
			id=figureId,
			type="figure",
			data={"payload": payload},
		))

	_addFigureEdges(graph, chunks, figuresById)
	_addRelatedChunkEdges(graph, chunks, relatedTopK, relatedThreshold, llm)
	if enableDependencyEdges:
		_addDependencyEdges(graph, chunks, dependencyTopK, dependencyThreshold, llm)

	return graph


def _addFigureEdges(graph: IngestionGraph, chunks: List[Chunk], figuresById: Dict[str, dict]):
	figureSet = set(figuresById.keys())
	for chunk in chunks:
		for figureId in chunk.figure_ids:
			if figureId in figureSet:
				graph.addEdge(GraphEdge(
					from_id=chunk.id,
					to_id=figureId,
					type="uses_figure",
					evidence="proposition_reference",
					description="Figure referenced by proposition text.",
				))

		for elementId in chunk.source_element_ids:
			if elementId in figureSet and elementId not in chunk.figure_ids:
				graph.addEdge(GraphEdge(
					from_id=chunk.id,
					to_id=elementId,
					type="uses_figure",
					evidence="source_element",
					description="Figure referenced by source element provenance.",
				))


def _addRelatedChunkEdges(
	graph: IngestionGraph,
	chunks: List[Chunk],
	relatedTopK: int,
	relatedThreshold: float,
	llm: Optional[LLM],
):
	similarityEdges = _collectSimilarityEdges(chunks, relatedTopK, relatedThreshold)
	for idx, otherIdx, score in similarityEdges:
		description = _describeRelation(chunks[idx], chunks[otherIdx], score, llm)
		graph.addEdge(GraphEdge(
			from_id=chunks[idx].id,
			to_id=chunks[otherIdx].id,
			type="related_to",
			evidence="semantic_similarity",
			description=description,
			kind="semantic_similarity",
			score=float(score),
		))

def _collectSimilarityEdges(
	chunks: List[Chunk],
	topK: int,
	threshold: float,
) -> List[Tuple[int, int, float]]:
	embeddings = [chunk.embedding for chunk in chunks]
	if len(chunks) <= topK + 1:
		return _collectSimilarityEdgesExact(chunks, embeddings, topK, threshold)
	return _collectSimilarityEdgesAnn(chunks, embeddings, topK, threshold)


def _collectSimilarityEdgesExact(
	chunks: List[Chunk],
	embeddings: List[Optional[List[float]]],
	topK: int,
	threshold: float,
) -> List[Tuple[int, int, float]]:
	edges: List[Tuple[int, int, float]] = []
	for idx, _chunk in enumerate(chunks):
		if embeddings[idx] is None:
			continue

		sims: List[Tuple[int, float]] = []
		for otherIdx, _ in enumerate(chunks):
			if idx == otherIdx or embeddings[otherIdx] is None:
				continue
			score = _cosineSimilarity(embeddings[idx], embeddings[otherIdx])
			sims.append((otherIdx, score))

		sims.sort(key=lambda item: item[1], reverse=True)
		for otherIdx, score in sims[:topK]:
			if score < threshold:
				continue
			edges.append((idx, otherIdx, score))
	return edges


def _collectSimilarityEdgesAnn(
	chunks: List[Chunk],
	embeddings: List[Optional[List[float]]],
	topK: int,
	threshold: float,
) -> List[Tuple[int, int, float]]:
	edges: List[Tuple[int, int, float]] = []
	buckets, planes, validIndices = _buildAnnIndex(
		embeddings,
		numTables=ANN_TABLES,
		numPlanes=ANN_PLANES,
		seed=ANN_SEED,
	)
	if not planes:
		return edges

	for idx, _chunk in enumerate(chunks):
		if embeddings[idx] is None or idx not in validIndices:
			continue

		candidates = _getAnnCandidates(idx, embeddings, buckets, planes)
		if not candidates:
			continue

		sims: List[Tuple[int, float]] = []
		for otherIdx in candidates:
			if embeddings[otherIdx] is None:
				continue
			score = _cosineSimilarity(embeddings[idx], embeddings[otherIdx])
			sims.append((otherIdx, score))

		sims.sort(key=lambda item: item[1], reverse=True)
		for otherIdx, score in sims[:topK]:
			if score < threshold:
				continue
			edges.append((idx, otherIdx, score))
	return edges


def _addDependencyEdges(
	graph: IngestionGraph,
	chunks: List[Chunk],
	dependencyTopK: int,
	dependencyThreshold: float,
	llm: Optional[LLM],
):
	similarityEdges = _collectSimilarityEdges(chunks, dependencyTopK, dependencyThreshold)
	for idx, otherIdx, score in similarityEdges:
		chunkA = chunks[idx]
		chunkB = chunks[otherIdx]
		direction, reason = _inferDependencyDirection(chunkA, chunkB, llm)
		if direction == "a_to_b":
			graph.addEdge(GraphEdge(
				from_id=chunkA.id,
				to_id=chunkB.id,
				type="requires",
				evidence="semantic_similarity",
				description=reason,
				kind="dependency",
				score=float(score),
			))
			continue
		if direction == "b_to_a":
			graph.addEdge(GraphEdge(
				from_id=chunkB.id,
				to_id=chunkA.id,
				type="requires",
				evidence="semantic_similarity",
				description=reason,
				kind="dependency",
				score=float(score),
			))
			continue
		graph.addEdge(GraphEdge(
			from_id=chunkA.id,
			to_id=chunkB.id,
			type="dependency_candidate",
			evidence="semantic_similarity",
			description="Semantic similarity suggests a potential dependency; direction unresolved.",
			kind="dependency_candidate",
			score=float(score),
		))


def _inferDependencyDirection(
	chunkA: Chunk,
	chunkB: Chunk,
	llm: Optional[LLM],
) -> Tuple[str, str]:
	if llm is None:
		return "unknown", "No dependency direction model provided."

	prompt = (
		"Determine prerequisite direction between two knowledge chunks.\n"
		"Return JSON with keys:\n"
		'{"direction": "A->B" | "B->A" | "none", "reason": "<short reason>"}\n\n'
		f"Chunk A propositions:\n{_formatPropositions(chunkA.data)}\n\n"
		f"Chunk B propositions:\n{_formatPropositions(chunkB.data)}\n"
	)
	response = llm.chat(
		messages=[{"role": "user", "content": prompt}],
		model="gpt-4o",
		response_format={"type": "json_object"},
	)
	try:
		data = json.loads(response)
	except json.JSONDecodeError:
		return "unknown", "Dependency direction parse failed."

	direction = str(data.get("direction", "")).strip().upper()
	reason = str(data.get("reason", "")).strip()
	if direction == "A->B":
		return "a_to_b", reason or "Chunk A is a prerequisite for Chunk B."
	if direction == "B->A":
		return "b_to_a", reason or "Chunk B is a prerequisite for Chunk A."
	return "none", reason or "No prerequisite relation identified."


# Builds ANN buckets for cosine similarity using random hyperplanes
def _buildAnnIndex(
	embeddings: List[Optional[List[float]]],
	numTables: int,
	numPlanes: int,
	seed: int,
) -> Tuple[List[Dict[int, List[int]]], List[np.ndarray], Set[int]]:
	validIndices = [idx for idx, emb in enumerate(embeddings) if emb is not None]
	if not validIndices:
		return [], [], set()

	dimension = len(embeddings[validIndices[0]])
	rng = np.random.RandomState(seed)
	planes = [
		rng.normal(size=(numPlanes, dimension)).astype(np.float32)
		for _ in range(numTables)
	]
	buckets = [defaultdict(list) for _ in range(numTables)]

	for idx in validIndices:
		vector = np.array(embeddings[idx], dtype=np.float32)
		for tableIdx, plane in enumerate(planes):
			signature = _hashSignature(plane, vector)
			buckets[tableIdx][signature].append(idx)

	return buckets, planes, set(validIndices)


# Returns ANN candidates from matching buckets
def _getAnnCandidates(
	idx: int,
	embeddings: List[Optional[List[float]]],
	buckets: List[Dict[int, List[int]]],
	planes: List[np.ndarray],
) -> Set[int]:
	vector = np.array(embeddings[idx], dtype=np.float32)
	candidates = set()
	for tableIdx, plane in enumerate(planes):
		signature = _hashSignature(plane, vector)
		candidates.update(buckets[tableIdx].get(signature, []))

	candidates.discard(idx)
	return candidates


# Hashes a vector against hyperplanes
def _hashSignature(planes: np.ndarray, vector: np.ndarray) -> int:
	projections = planes @ vector
	signature = 0
	for value in projections:
		signature = (signature << 1) | (1 if value >= 0 else 0)
	return signature


def _cosineSimilarity(vecA: List[float], vecB: List[float]) -> float:
	arrA = np.array(vecA, dtype=np.float32)
	arrB = np.array(vecB, dtype=np.float32)
	denom = (np.linalg.norm(arrA) * np.linalg.norm(arrB))
	if denom == 0.0:
		return 0.0
	return float(np.dot(arrA, arrB) / denom)


def _describeRelation(chunkA: Chunk, chunkB: Chunk, score: float, llm: Optional[LLM]) -> str:
	if llm is None:
		return f"Semantic similarity score {score:.2f} between chunks."

	prompt = (
		"Write one concise sentence describing how the two chunks are related.\n\n"
		f"Chunk A:\n{_formatPropositions(chunkA.data)}\n\n"
		f"Chunk B:\n{_formatPropositions(chunkB.data)}\n"
	)
	try:
		response = llm.chat(
			messages=[{"role": "user", "content": prompt}],
			model="gpt-4o-mini",
		)
		return response.strip()
	except Exception:
		return f"Semantic similarity score {score:.2f} between chunks."


def _formatPropositions(propositions: List[str]) -> str:
	return "\n".join(f"- {prop}" for prop in propositions[:8])
