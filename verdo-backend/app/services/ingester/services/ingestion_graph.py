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
	if len(chunks) <= relatedTopK + 1:
		_addRelatedChunkEdgesExact(graph, chunks, relatedTopK, relatedThreshold, llm)
		return

	_addRelatedChunkEdgesAnn(graph, chunks, relatedTopK, relatedThreshold, llm)


# Builds related_to edges using full pairwise comparisons for small corpora
def _addRelatedChunkEdgesExact(
	graph: IngestionGraph,
	chunks: List[Chunk],
	relatedTopK: int,
	relatedThreshold: float,
	llm: Optional[LLM],
):
	embeddings = [chunk.embedding for chunk in chunks]
	for idx, chunk in enumerate(chunks):
		if embeddings[idx] is None:
			continue

		sims = []
		for otherIdx, _ in enumerate(chunks):
			if idx == otherIdx or embeddings[otherIdx] is None:
				continue
			score = _cosineSimilarity(embeddings[idx], embeddings[otherIdx])
			sims.append((otherIdx, score))

		_addRelatedCandidates(graph, chunk, chunks, sims, relatedTopK, relatedThreshold, llm)


# Builds related_to edges using ANN candidate selection
def _addRelatedChunkEdgesAnn(
	graph: IngestionGraph,
	chunks: List[Chunk],
	relatedTopK: int,
	relatedThreshold: float,
	llm: Optional[LLM],
):
	embeddings = [chunk.embedding for chunk in chunks]
	buckets, planes, validIndices = _buildAnnIndex(
		embeddings,
		numTables=ANN_TABLES,
		numPlanes=ANN_PLANES,
		seed=ANN_SEED,
	)
	if not planes:
		return

	for idx, chunk in enumerate(chunks):
		if embeddings[idx] is None or idx not in validIndices:
			continue

		candidates = _getAnnCandidates(idx, embeddings, buckets, planes)
		if not candidates:
			continue

		sims = []
		for otherIdx in candidates:
			if embeddings[otherIdx] is None:
				continue
			score = _cosineSimilarity(embeddings[idx], embeddings[otherIdx])
			sims.append((otherIdx, score))

		_addRelatedCandidates(graph, chunk, chunks, sims, relatedTopK, relatedThreshold, llm)


# Creates related_to edges from scored candidates
def _addRelatedCandidates(
	graph: IngestionGraph,
	chunk: Chunk,
	chunks: List[Chunk],
	sims: List[Tuple[int, float]],
	relatedTopK: int,
	relatedThreshold: float,
	llm: Optional[LLM],
):
	sims.sort(key=lambda item: item[1], reverse=True)
	for otherIdx, score in sims[:relatedTopK]:
		if score < relatedThreshold:
			continue
		description = _describeRelation(chunk, chunks[otherIdx], score, llm)
		graph.addEdge(GraphEdge(
			from_id=chunk.id,
			to_id=chunks[otherIdx].id,
			type="related_to",
			evidence="semantic_similarity",
			description=description,
			kind="semantic_similarity",
			score=float(score),
		))


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
