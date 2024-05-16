"""
Generate summaries of the explanation templates using LLMs.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B-Instruct",
    # token='hf_xxx'
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    local_files_only=True
)

def load_data(
        templates: list,
    ) -> dict:

    for template in templates:
        path = Path(f'resources/datasets/{template}.json')
        if not path.exists():
            raise FileNotFoundError(f'The template {template} does not exist.')
        with path.open(encoding='utf-8') as f:
            data = json.load(f)

    return data


def format_data(
        data: dict,
    ) -> dict:

    sys_message = "The following text contains a question, followed by a process that was used to solve it. Select the steps in the process that are most relevant for inclusion in a summary of that process. Format your output using a list containing the indices of the selected steps, for example: [1, 2, 4]."

    inputs = []
    for example in data.values():

        input_text = ""
        input_text += f"Question: {example['question']}. Process: "
        for step in example['explanation']:
            input_text += f"{step['step']}. {step['explanation']}. "

        inputs.append(input_text)

    prompts = [f"<s>[INST] <<SYS>>{sys_message}<</SYS>>{i} [/INST]" for i in inputs]

    return prompts


def generate_summaries(
        inputs: list,
    ) -> list:

    input_ids = [tokenizer.encode(i, return_tensors="pt") for i in inputs]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    outputs = model.generate(input_ids)
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return responses


def construct_results(
        data: dict,
        responses: list,
    ) -> dict:

    results = pd.DataFrame(columns = list(data.keys()), data=[responses])

    return results

def main(
        args
    ) -> None:

    data = load_data(args.template)
    inputs = format_data(data)
    responses = generate_summaries(inputs)
    results = construct_results(data, responses)

    return inputs



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate summaries of the explanation templates using LLMs.')
    parser.add_argument('-t', '--template', nargs='+', required=True, type=str, help='The name of the explanation template.')
    args = parser.parse_args()

