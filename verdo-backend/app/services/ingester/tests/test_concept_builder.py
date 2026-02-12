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

from app.services.ingester.services.chunker import Chunker
from app.services.ingester.services.ConceptBuilder import ConceptBuilder
from app.services.ingester.services.ingestion_graph import buildGraph

# --- Main ---------------------------------------------------------------

def main():
	# Path to Module7.json
	inputFile = PROJECT_ROOT / "out" / "Module7.json"
	outputFile = PROJECT_ROOT / "out" / "Module7_concepts_v2.json"
	propositionsFile = PROJECT_ROOT / "out" / "Module7_propositions.json"

	if not inputFile.exists():
		print(f"Error: Input file not found at {inputFile}")
		return

	print(f"Loading {inputFile}...")
	with open(inputFile, 'r', encoding='utf-8') as f:
		data = json.load(f)

	# Initialize Chunker
	chunker = Chunker()

	# Step 1: Extract elements
	print("\n📄 Step 1: Extracting elements...")
	chunker.getElements(data)
	print(f"   Extracted {len(chunker.elements)} elements")
	print(f"   Found {len(chunker.figuresById)} figures")

	# Step 2: Batch elements
	print("\n📦 Step 2: Batching elements...")
	chunker.batch(threshold=1000)
	print(f"   Created {len(chunker.batches)} batches")

	# Step 3: Generate propositions
	print("\n💬 Step 3: Generating propositions...")
	chunker.getPropositions()
	print(f"   Generated {len(chunker.propositions)} propositions")

	print(f"\n💾 Saving propositions to {propositionsFile}...")
	with open(propositionsFile, "w", encoding="utf-8") as f:
		json.dump(
			{
				"propositions": chunker.propositions,
				"count": len(chunker.propositions),
			},
			f,
			indent=2,
			ensure_ascii=False,
		)

	# Preview a few propositions
	print("\n   Sample propositions:")
	for i, prop in enumerate(chunker.propositions[:3]):
		propText = prop["text"] if isinstance(prop, dict) else prop
		preview = propText[:80] + "..." if len(propText) > 80 else propText
		print(f"   [{i}] {preview}")

	# Step 4: Build concepts using ConceptBuilder
	print("\n🧠 Step 4: Building concepts...")
	conceptBuilder = ConceptBuilder(printLogging=True)
	concepts = conceptBuilder.buildConcepts(chunker.propositions)

	# Print stats
	stats = conceptBuilder.getStats()
	print(f"\n📊 Statistics:")
	print(f"   Total concepts: {stats['totalConcepts']}")
	print(f"   Total propositions: {stats['totalPropositions']}")
	print(f"   Avg propositions per concept: {stats.get('avgPropsPerConcept', 0):.1f}")
	print(f"   Min propositions per concept: {stats.get('minProps', 0)}")
	print(f"   Max propositions per concept: {stats.get('maxProps', 0)}")

	# Print concepts
	print(f"\n📚 Concepts ({len(concepts)}):")
	print("=" * 70)
	for conceptId, concept in concepts.items():
		print(f"\n🔹 {concept.title}")
		print(f"   ID: {conceptId}")
		print(f"   Summary: {concept.summary}")
		print(f"   Propositions ({len(concept.propositions)}):")
		for prop in concept.propositions[:3]:  # Show first 3
			preview = prop[:60] + "..." if len(prop) > 60 else prop
			print(f"      - {preview}")
		if len(concept.propositions) > 3:
			print(f"      ... and {len(concept.propositions) - 3} more")

	# Step 5: Build concept DAG
	print("\n🕸️ Step 5: Building concept DAG...")
	graph = buildGraph(concepts, strict=True, autoDropCycleEdges=False)
	print(f"   Graph nodes: {len(graph.nodes)}")
	print(f"   Graph edges: {len(graph.edges)}")
	order = graph.topologicalOrder()
	print(f"   Topological order length: {len(order)}")

	graphPath = PROJECT_ROOT / "out" / "Module7_concept_graph.dot"
	_write_graphviz(graph, graphPath)
	print(f"   Wrote graphviz output to {graphPath}")

	# Save output
	print(f"\n💾 Saving to {outputFile}...")
	output = {
		"stats": stats,
		"concepts": conceptBuilder.getConceptsList(),
		"graph": {
			"nodes": [vars(node) for node in graph.nodes.values()],
			"edges": [vars(edge) for edge in graph.edges],
			"topologicalOrder": order,
		},
	}
	with open(outputFile, 'w', encoding='utf-8') as f:
		json.dump(output, f, indent=2, ensure_ascii=False)
	
	print(f"✅ Done! Saved {len(concepts)} concepts to {outputFile}")

	return concepts


def _write_graphviz(graph, path: Path) -> None:
	lines = ["digraph ConceptGraph {"]
	lines.append('  rankdir="LR";')
	lines.append('  node [shape=box, fontsize=10];')

	for node_id, node in graph.nodes.items():
		title = node.data.get("title", "") if isinstance(node.data, dict) else ""
		label = f"{title}\\n{node_id}" if title else node_id
		lines.append(f'  "{node_id}" [label="{label}"];')

	for edge in graph.edges:
		lines.append(
			f'  "{edge.from_id}" -> "{edge.to_id}" '
			f'[label="{edge.type}", fontsize=9];'
		)

	lines.append("}")
	path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
	main()
