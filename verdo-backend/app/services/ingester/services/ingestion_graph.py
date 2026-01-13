from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from app.services.ingester.services.LLM import LLM
from app.services.ingester.services.chunk import Chunk


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
	embeddings = [chunk.embedding for chunk in chunks]
	for idx, chunk in enumerate(chunks):
		if embeddings[idx] is None:
			continue
		sims = []
		for otherIdx, otherChunk in enumerate(chunks):
			if idx == otherIdx or embeddings[otherIdx] is None:
				continue
			score = _cosineSimilarity(embeddings[idx], embeddings[otherIdx])
			sims.append((otherIdx, score))

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
