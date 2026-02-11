import os

# Helper to load prompt content
def _load_prompt(filename):
	current_dir = os.path.dirname(os.path.abspath(__file__))
	try:
		with open(os.path.join(current_dir, filename), 'r', encoding='utf-8') as f:
			return f.read().strip()
	except FileNotFoundError:
		return ""

# Load prompts into constants
DESCRIBE_IMAGE_PROMPT = _load_prompt('describe_image.txt')
EXTRACT_TABLE_PROMPT = _load_prompt('extract_table.txt')
EXTRACT_EQUATION_IMAGE_PROMPT = _load_prompt('extract_equation_image.txt')
EXTRACT_EQUATION_TEXT_PROMPT = _load_prompt('extract_equation_text.txt')
DECOMPOSE_PROPOSITIONS_PROMPT = _load_prompt('decompose_propositions.txt')
CONCEPT_DECISIONS_SYSTEM_PROMPT = _load_prompt('concept_decisions_system.txt')
CONCEPT_DECISIONS_USER_PROMPT = _load_prompt('concept_decisions_user.txt')
UPDATE_SUMMARY_SYSTEM_PROMPT = _load_prompt('update_summary_system.txt')
UPDATE_SUMMARY_USER_PROMPT = _load_prompt('update_summary_user.txt')

# New Categorization Prompts
CATEGORIZE_IMAGE_PROMPT = _load_prompt('categorize_image.txt')
DESCRIBE_DIAGRAM_PROMPT = _load_prompt('describe_diagram.txt')
DESCRIBE_CHART_PROMPT = _load_prompt('describe_chart.txt')
DESCRIBE_PHOTO_PROMPT = _load_prompt('describe_photo.txt')
DESCRIBE_FORMULA_PROMPT = _load_prompt('describe_formula.txt')
DESCRIBE_TABLE_PROMPT = _load_prompt('describe_table.txt')
DESCRIBE_TEXT_IMAGE_PROMPT = _load_prompt('describe_text_image.txt')
DESCRIBE_FLOWCHART_PROMPT = _load_prompt('describe_flowchart.txt')

# Concept Builder Prompts
CONCEPT_CREATION_SYSTEM_PROMPT = _load_prompt('concept_creation_system.txt')
CONCEPT_CREATION_USER_PROMPT = _load_prompt('concept_creation_user.txt')
CONCEPT_MERGE_SYSTEM_PROMPT = _load_prompt('concept_merge_system.txt')
CONCEPT_MERGE_USER_PROMPT = _load_prompt('concept_merge_user.txt')
