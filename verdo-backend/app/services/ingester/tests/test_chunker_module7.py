import json
import os
import sys
from pathlib import Path

# --- Setup --------------------------------------------------------------

# Resolve project root (expects 'app' dir present up the tree)
PROJECT_ROOT = Path(__file__).resolve()
for _ in range(8):
	if (PROJECT_ROOT / 'app').exists():
		break
	PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

try:
	from dotenv import load_dotenv
	load_dotenv(PROJECT_ROOT / "app" / "services" / ".env")
except Exception:
	pass

try:
	from app.services.ingester.services.chunker import Chunker
	from app.services.ingester.services.ingestion_graph import buildGraph
except ImportError:
	# Fallback for direct execution if package structure varies
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from app.services.ingester.services.chunker import Chunker
	from app.services.ingester.services.ingestion_graph import buildGraph

# --- Main ---------------------------------------------------------------

def main():
	# Path to Module7.json
	inputFile = PROJECT_ROOT / "out" / "Module7.json"
	outputFile = PROJECT_ROOT / "out" / "Module7_chunks.json"

	if not inputFile.exists():
		print(f"Error: Input file not found at {inputFile}")
		return

	print(f"Loading {inputFile}...")
	with open(inputFile, 'r', encoding='utf-8') as f:
		data = json.load(f)

	# Extract propositions (content from elements)
	propositions = []
	for page in data:
		for element in page.get('elements', []):
			content = str(element.get('content', '')).strip()
			if content:
				propositions.append(content)

	print(f"Extracted {len(propositions)} propositions.")

	# Initialize Chunker
	chunker = Chunker()

	# Run Chunker
	print("Running AgenticChunker...")
	chunker.getElements(data)
	print(f"Extracted {len(chunker.elements)} elements.")

	chunker.batch(threshold=1000)
	print(f"Created {len(chunker.batches)} batches.")

	print("Generating propositions...")
	chunker.getPropositions()

	print(f"Generated {len(chunker.propositions)} propositions.")
	
	print("Embedding propositions...")
	chunker.embedPropositions()
	
	print("Clustering propositions...")
	chunker.clusterPropositions()
	
	print("Building semantic chunks...")
	chunker.buildSemanticChunks()
	
	print("Building concept graph...")
	graph = buildGraph(chunker.semanticChunks, chunker.figuresById)
	print(f"Graph nodes: {len(graph.nodes)}, edges: {len(graph.edges)}")

	print(f"\n--- Semantic Chunks ({len(chunker.semanticChunks)}) ---\n")
	
	for chunk in chunker.semanticChunks:
		print(f"Chunk {chunk.id}:")
		for prop in chunk.data:
			print(f"- {prop}")
		print("") # clear line between clusters

	return graph

if __name__ == "__main__":
	main()
