from pathlib import Path

MODEL = 'nfs/public/hf/models/meta-llama/Meta-Llama-3-8B-Instruct'
TEMPERATURE = 0.3
EXAMPLE = '[1, 2, 3, 4]'
FORMAT_VERSION = 2
SYSTEM_CONTENT = f"Your role is to select, from a list the steps, those that are most important for inclusion in a summary explanation of that process. Format your output as a list, for example {EXAMPLE}. Output only this short summary paragraph and nothing else."
INPUTS = 'select.json'