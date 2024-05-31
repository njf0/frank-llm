import argparse
import datetime
import json
import logging
import random
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger('transformers').setLevel(logging.ERROR)

class LLMParticipant:

    def __init__(
            self,
            config: dict
        ) -> None:

        self.config = config
        self.MODEL = config['MODEL']
        self.model_family = self.MODEL.split('/')[0]
        self.TEMPERATURE = config['TEMPERATURE']
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.EXAMPLE = config['EXAMPLE']
        self.shot = config['SHOT']
        self.INPUT = config['INPUT']
        self.SYSTEM_CONTENT = config['SYSTEM_CONTENT']
        self.FORMAT_VERSION = config['FORMAT_VERSION']

        self.data = self.load_input_questions()

        print(f"Loading model {self.MODEL}... ", end="")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL,
            # token='hf_xxx'
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # local_files_only=True
        )
        print("done.")

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

    def load_input_questions(
            self,
        ) -> dict:

        with Path(self.INPUT).open(encoding='utf-8') as f:
            data = json.load(f)

        return data

    def assemble_messages(
            self,
            data: dict,
        ) -> list[list[dict]]:

        inputs = []
        for example in data.values():

            user_content = ""
            user_content += f"Question: {example['question']} Process: "
            for step in example['explanation']:
                user_content += f"{step['step']}. {step['explanation']}. "
            user_content += f"List of selected steps for inclusion in summary: ["

            if self.model_family in ['']:

                messages = [
                    {
                        "role": "system",
                        "content": self.SYSTEM_CONTENT
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ]

            elif self.model_family in ['mistralai', 'microsoft']:
                messages = [
                    {
                        "role": "user",
                        "content": self.SYSTEM_CONTENT + ' ' + user_content
                    },
                ]

            elif self.model_family in ['google']:
                messages = f'{self.SYSTEM_CONTENT} {user_content}'

            inputs.append(messages)

        return inputs

    def apply_template_generate_response(
            self,
            inputs
        ) -> list[str]:

        responses = []
        batched_inputs = [inputs[i:i+self.BATCH_SIZE] for i in range(0, len(inputs), self.BATCH_SIZE)]

        for batch in batched_inputs:

            if self.model_family in ['', 'mistralai', 'microsoft']:
                input_messages = self.tokenizer.apply_chat_template(batch,
                                                                    padding=True,
                                                                    return_tensors="pt",
                                                                    tokenize=True,
                                                                    add_generation_prompt=True).to(self.device)
            elif self.model_family in ['google']:
                input_messages = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)['input_ids']
            outputs = self.model.generate(input_messages,
                                          max_new_tokens=256,
                                          do_sample=True,
                                          temperature=self.TEMPERATURE
                                          )
            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return responses

    def save_results(
            self,
            results: list[dict]
        ):

        str_shot_dict = {0: 'zero', 1: 'one', 3: 'few'}
        output_filename = datetime.datetime.isoformat(datetime.datetime.now())
        output_path = Path("outputs",
                           f"{Path(self.INPUT).stem}",
                           f"{str_shot_dict[self.config['shot']]}",
                           f"{output_filename}.json")
        log = Path('log.json')
        print(f"Saving results in {output_path}... ", end="")

        outputs = {
            "config": self.config,
            "results": results
        }

        with output_path.open('w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=4)
        print("done.")

        with log.open('r', encoding='utf-8') as f:
            log_data = json.load(f)

        log_data[output_filename] = self.config
        # sort log
        log_data = dict(sorted(log_data.items()))

        with log.open('w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4)

    def participate(
            self
        ):

        print(f"Assembling messages for {len(self.data)} inputs... ", end="")
        inputs = self.assemble_messages(self.data)
        print("done.")
        print(f"Generating responses for {len(inputs)} inputs... ")
        responses = self.apply_template_generate_response(inputs)
        print("Done.")
        outputs = []
        for q, r in zip(self.data.keys(), responses):
            t = self.data[q]['template_id']
            outputs.append({
                "question_id": q,
                "template_id": t,
                "response": r,
            })

        self.save_results(outputs)

        return outputs





if __name__ == '__main__':


    MODEL = 'microsoft/Phi-3-small-128k-instruct'
    TEMPERATURE = 0.3
    BATCH_SIZE = 20
    EXAMPLE = '[1, 2, 3, 4]'
    FORMAT_VERSION = 2
    SYSTEM_CONTENT = f"Your role is to select, from a list the steps, only those that are most important for inclusion in a summary explanation of that process. Format your output as a list, for example {EXAMPLE}. Output only this short summary paragraph and nothing else.".format(EXAMPLE)
    INPUTS = '../resources/data/select.json'
    DESCRIPTION = "testing"

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=str, default='queue.json')
    parser.add_argument('--model', type=str)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--example', type=str)
    parser.add_argument('--system_content', type=str)
    parser.add_argument('--inputs', type=str)
    parser.add_argument('--description', type=str)
    args = parser.parse_args()

    if not args.batch:
        config = {
            "MODEL": MODEL,
            "TEMPERATURE": TEMPERATURE,
            "BATCH_SIZE": BATCH_SIZE,
            "EXAMPLE": EXAMPLE,
            "FORMAT_VERSION": FORMAT_VERSION,
            "SYSTEM_CONTENT": SYSTEM_CONTENT,
            "INPUTS": INPUTS,
            "DESCRIPTION": DESCRIPTION
        }

    else:
        with Path(args.batch).open(encoding='utf-8') as f:
            configs = json.load(f)

        completed = []
        for description, config in configs.items():
            print(f"Processing {description}...")
            participant = LLMParticipant(config)
            participant.participate()
            completed.append(config['DESCRIPTION'])
            print(f"Completed {description}.")

        # remove completed configs from config file
        for config in completed:
            configs.remove(config)

        with Path(args.batch).open('w', encoding='utf-8') as f:
            json.dump(configs, f, indent=4)
