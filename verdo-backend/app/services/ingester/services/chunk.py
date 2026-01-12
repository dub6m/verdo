from typing import Dict, List, Optional


class Chunk:
	def __init__(
		self,
		chunk_id: str,
		data: List[str],
		confidence: float = 0.0,
		embedding: Optional[List[float]] = None,
		figure_ids: Optional[List[str]] = None,
		source_element_ids: Optional[List[str]] = None,
		relations: Optional[Dict[str, List[str]]] = None,
	):
		self.id = chunk_id
		self.data = data
		self.confidence = confidence
		self.embedding = embedding
		self.figure_ids = figure_ids or []
		self.source_element_ids = source_element_ids or []
		self.relations = relations or {}
