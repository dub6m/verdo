from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


class GraphValidationError(ValueError):
	pass


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

	def removeEdge(self, edge: GraphEdge):
		edgeId = edge.edgeId()
		if edgeId in self._edgeIds:
			self._edgeIds.remove(edgeId)
		self.edges = [existing for existing in self.edges if existing.edgeId() != edgeId]

	def topologicalOrder(self) -> List[str]:
		adj: Dict[str, List[str]] = {nodeId: [] for nodeId in self.nodes}
		inDegree: Dict[str, int] = {nodeId: 0 for nodeId in self.nodes}

		for edge in self.edges:
			if edge.from_id in adj and edge.to_id in inDegree:
				adj[edge.from_id].append(edge.to_id)
				inDegree[edge.to_id] += 1

		queue = [nodeId for nodeId, degree in inDegree.items() if degree == 0]
		order: List[str] = []
		idx = 0
		while idx < len(queue):
			nodeId = queue[idx]
			idx += 1
			order.append(nodeId)
			for neighbor in adj[nodeId]:
				inDegree[neighbor] -= 1
				if inDegree[neighbor] == 0:
					queue.append(neighbor)

		if len(order) != len(self.nodes):
			raise GraphValidationError("Cannot compute topological order: graph contains a cycle.")
		return order


@dataclass
class _ConceptRecord:
	id: str
	title: str
	summary: str
	propositions: List[str]
	propositionIndices: List[int]
	prerequisites: List[str]


def buildConceptGraph(
	concepts: Any,
	strict: bool = True,
	autoDropCycleEdges: bool = False,
) -> IngestionGraph:
	"""
	Build concept DAG where edges point from prerequisite -> dependent concept.

	Args:
		concepts: Dict[str, Concept|dict] or List[Concept|dict]
		strict: Fail fast on invalid references/cycles when True.
		autoDropCycleEdges: If True, removes one edge per detected cycle until acyclic.
	"""
	if not isinstance(strict, bool):
		raise TypeError("strict must be a bool.")
	if not isinstance(autoDropCycleEdges, bool):
		raise TypeError("autoDropCycleEdges must be a bool.")

	records = _normalizeConcepts(concepts)
	graph = IngestionGraph()

	for record in records:
		graph.addNode(GraphNode(
			id=record.id,
			type="concept",
			data={
				"title": record.title,
				"summary": record.summary,
				"propositions": record.propositions,
			},
			metadata={
				"proposition_indices": record.propositionIndices,
				"prerequisites": record.prerequisites,
			},
		))

	unknownErrors: List[str] = []
	validEdges: List[Tuple[str, str]] = []
	knownIds = set(graph.nodes.keys())
	for record in records:
		for prerequisiteId in record.prerequisites:
			if prerequisiteId not in knownIds:
				unknownErrors.append(
					f"Concept '{record.id}' references unknown prerequisite '{prerequisiteId}'."
				)
				continue
			validEdges.append((prerequisiteId, record.id))

	if unknownErrors and strict:
		raise GraphValidationError("\n".join(unknownErrors))

	for prereqId, conceptId in validEdges:
		graph.addEdge(GraphEdge(
			from_id=prereqId,
			to_id=conceptId,
			type="requires",
			evidence="concept_prerequisites",
			description=f"{conceptId} depends on prerequisite {prereqId}.",
			kind="dependency",
		))

	if autoDropCycleEdges:
		_dropCycleEdges(graph)
	else:
		cycle = _findCycle(graph)
		if cycle and strict:
			prettyCycle = " -> ".join(cycle)
			raise GraphValidationError(f"Cycle detected in concept prerequisites: {prettyCycle}")

	return graph


def buildGraph(
	concepts: Any,
	strict: bool = True,
	autoDropCycleEdges: bool = False,
) -> IngestionGraph:
	"""Backward-compatible alias for concept graph construction."""
	return buildConceptGraph(concepts, strict=strict, autoDropCycleEdges=autoDropCycleEdges)


def _normalizeConcepts(concepts: Any) -> List[_ConceptRecord]:
	if isinstance(concepts, dict):
		iterable: Iterable[Any] = concepts.values()
	elif isinstance(concepts, list):
		iterable = concepts
	else:
		raise TypeError("concepts must be a dict or list.")

	records: List[_ConceptRecord] = []
	for item in iterable:
		record = _normalizeConcept(item)
		records.append(record)

	ids = [record.id for record in records]
	duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
	if duplicates:
		raise GraphValidationError(f"Duplicate concept ids are not allowed: {', '.join(duplicates)}")

	return records


def _normalizeConcept(item: Any) -> _ConceptRecord:
	if isinstance(item, dict):
		conceptId = str(item.get("id", "")).strip()
		title = str(item.get("title", "")).strip()
		summary = str(item.get("summary", "")).strip()
		propositions = _coerceList(item.get("propositions", []), str)
		indices = _coerceList(item.get("propositionIndices", []), int)
		prerequisites = _coerceList(item.get("prerequisites", []), str)
	else:
		conceptId = str(getattr(item, "id", "")).strip()
		title = str(getattr(item, "title", "")).strip()
		summary = str(getattr(item, "summary", "")).strip()
		propositions = _coerceList(getattr(item, "propositions", []), str)
		indices = _coerceList(getattr(item, "propositionIndices", []), int)
		prerequisites = _coerceList(getattr(item, "prerequisites", []), str)

	if not conceptId:
		raise GraphValidationError("Every concept must have a non-empty id.")

	return _ConceptRecord(
		id=conceptId,
		title=title,
		summary=summary,
		propositions=propositions,
		propositionIndices=indices,
		prerequisites=prerequisites,
	)


def _coerceList(value: Any, castType):
	if value is None:
		return []
	if not isinstance(value, list):
		raise GraphValidationError(f"Expected list value, got {type(value).__name__}.")
	result = []
	for item in value:
		if castType is int:
			result.append(int(item))
		else:
			result.append(str(item))
	return result


def _findCycle(graph: IngestionGraph) -> List[str]:
	adj: Dict[str, List[str]] = {nodeId: [] for nodeId in graph.nodes}
	for edge in graph.edges:
		if edge.from_id in adj and edge.to_id in graph.nodes:
			adj[edge.from_id].append(edge.to_id)

	state: Dict[str, int] = {nodeId: 0 for nodeId in graph.nodes}
	stack: List[str] = []
	indexByNode: Dict[str, int] = {}

	def dfs(nodeId: str) -> List[str]:
		state[nodeId] = 1
		indexByNode[nodeId] = len(stack)
		stack.append(nodeId)

		for neighbor in adj[nodeId]:
			if state[neighbor] == 0:
				cycle = dfs(neighbor)
				if cycle:
					return cycle
			elif state[neighbor] == 1:
				start = indexByNode[neighbor]
				return stack[start:] + [neighbor]

		stack.pop()
		indexByNode.pop(nodeId, None)
		state[nodeId] = 2
		return []

	for nodeId in graph.nodes:
		if state[nodeId] == 0:
			cycle = dfs(nodeId)
			if cycle:
				return cycle

	return []


def _dropCycleEdges(graph: IngestionGraph):
	while True:
		cycle = _findCycle(graph)
		if not cycle:
			return

		if len(cycle) < 2:
			raise GraphValidationError("Cycle detection failed to identify removable edge.")

		removalFrom = cycle[-2]
		removalTo = cycle[-1]
		edgeToRemove = None
		for edge in graph.edges:
			if edge.from_id == removalFrom and edge.to_id == removalTo and edge.type == "requires":
				edgeToRemove = edge
				break

		if edgeToRemove is None:
			raise GraphValidationError(
				f"Unable to remove cycle edge {removalFrom} -> {removalTo}; edge not found."
			)
		graph.removeEdge(edgeToRemove)
