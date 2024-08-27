import argparse
import json
import logging
import random

import pandas as pd
import torch
from base import GenerationBase
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# FRANKLIN: Frank Library of Ideal Narratives (?!)

logging.getLogger('transformers').setLevel(logging.ERROR)

class MetaLlama(GenerationBase):
    """
    Implementation of Llama chat templates etc for application to dataset.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model and data parameters.
    """
    def __init__(
            self,
            config: dict,
    ) -> None:

        super().__init__(config)

        assert config['model'].split('/')[0] == 'meta-llama', 'Model must be a Meta-Llama model.'

        local_path = '/nfs/public/hf/models/'
        full_model_path = local_path + config['model']

        self.tokenizer = self.tokenizer = AutoTokenizer.from_pretrained(
            full_model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            full_model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.device = self.model.device
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def assemble_messages(
            self,
            data: pd.DataFrame,
    ) -> list:
        """
        Assemble messages for input to model.

        Parameters
        ----------
        inputs : list
            List of input questions.

        Returns
        -------
        messages : list
            List of messages for input to model.
        """
        messages = []

        for i in tqdm(data["question"], desc="Assembling messages"):
            messages.append(
                [
                    {"role": "system", "content": self.config["system_content"]},
                    {"role": "user", "content": i},
                ]
            )

        data["messages"] = messages

        return data

    def apply_and_generate(
            self,
            data: pd.DataFrame,
    ) -> list[str]:
        """
        Apply chat templates and generate responses.

        Parameters
        ----------
        messages : list
            List of messages for input to model.

        Returns
        -------
        responses : list
            List of generated responses.
        """
        responses = []
        messages = data["messages"].tolist()

        batched_inputs = [
            messages[i : i + self.config["batch_size"]]
            for i in range(0, len(messages), self.config["batch_size"])
        ]

        for batch in tqdm(batched_inputs, desc="Generating batch responses"):

            inputs = self.tokenizer.apply_chat_template(
                batch,
                padding=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=self.config["temperature"],
            )

            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        data["responses"] = responses

        return data

    def parse_outputs(
            self,
            data: pd.DataFrame,
    ) -> list:
        """
        Parse outputs from model.

        Parameters
        ----------
        outputs : list
            List of generated responses.

        Returns
        -------
        responses : list
            List of parsed responses.
        """
        generation_prompt = "assistant\n\n"

        responses = []

        for response in tqdm(data["responses"], desc="Parsing responses"):
            responses.append(response.split(generation_prompt)[-1].strip())

        data["parsed_responses"] = responses

        return data

    def run(
            self,
            save: bool = True,
    ) -> dict:
        """
        Run the model on the dataset.
        """
        df = self.load_inputs()[:self.config["examples"]]
        df = self.assemble_messages(df)
        df = self.apply_and_generate(df)
        df = self.parse_outputs(df)
        if save:
            self.save_results(df)

        return

class Mistral(GenerationBase):
    """
    Implementation of Mistral chat templates etc for application to dataset.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model and data parameters.
    """
    def __init__(
            self,
            config: dict,
    ) -> None:

        super().__init__(config)

        # check model is a Mistral model
        if config['model'].split('/')[0] != 'mistralai':
            raise ValueError("Model(/family) doesn't appear correct.")

        self.tokenizer = self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"],
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model"],
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.device = self.model.device
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def assemble_messages(
            self,
            inputs: list[str],
    ) -> list:
        """
        Assemble messages for input to model.

        Parameters
        ----------
        inputs : list
            List of input questions.

        Returns
        -------
        messages : list
            List of messages for input to model.
        """
        messages = []

        for i in tqdm(inputs, desc="Assembling messages"):
            messages.append(
                [
                    {
                        "role": "user",
                        "content": self.config['system_content'] + ' ' + i,
                    },
                ]
            )

        return messages

    def apply_and_generate(
            self,
            messages: list[str],
    ) -> list[str]:
        """
        Apply chat templates and generate responses.

        Parameters
        ----------
        messages : list
            List of messages for input to model.

        Returns
        -------
        responses : list
            List of generated responses.
        """
        responses = []

        batched_inputs = [
            messages[i : i + self.config["batch_size"]]
            for i in range(0, len(messages), self.config["batch_size"])
        ]

        for batch in tqdm(batched_inputs, desc="Generating batch responses"):

            inputs = self.tokenizer.apply_chat_template(
                batch,
                padding=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=self.config["temperature"],
            )

            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return responses

    def parse_outputs(
            self,
            inputs: list[str],
            outputs: list[str],
    ) -> list:
        """
        Parse outputs from model.

        Parameters
        ----------
        outputs : list
            List of generated responses.

        Returns
        -------
        responses : list
            List of parsed responses.
        """
        responses = []

        for in_out_pair in tqdm(zip(inputs, outputs), desc="Parsing outputs"):
            # responses.append(output[output.index(generation_prompt) + len(generation_prompt):])
            responses.append(in_out_pair[1].split(in_out_pair[0])[-1].strip())

        return responses

    def run(
            self,
            save: bool = True,
    ) -> dict:
        """
        Run the model on the dataset.
        """
        inputs = self.load_inputs()
        messages = self.assemble_messages(inputs)
        outputs = self.apply_and_generate(messages)
        responses = self.parse_outputs(inputs, outputs)
        if save:
            self.save_results(inputs, responses)

        return list(zip(inputs, responses))


class MicrosoftPhi(GenerationBase):
    """
    Implementation of Microsoft chat templates etc for application to dataset.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model and data parameters.
    """
    def __init__(
            self,
            config: dict,
    ) -> None:

        super().__init__(config)

        # check model is a Mistral model
        if config['model'].split('/')[0] != 'microsoft':
            raise ValueError("Model(/family) doesn't appear correct.")

        self.tokenizer = self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"],
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model"],
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.device = self.model.device
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def assemble_messages(
            self,
            inputs: list[str],
    ) -> list:
        """
        Assemble messages for input to model.

        Parameters
        ----------
        inputs : list
            List of input questions.

        Returns
        -------
        messages : list
            List of messages for input to model.
        """
        messages = []

        for i in tqdm(inputs, desc="Assembling messages"):
            messages.append(
                [
                    {
                        "role": "user",
                        "content": self.config['system_content'] + ' ' + i,
                    },
                ]
            )

        return messages

    def apply_and_generate(
            self,
            messages: list[str],
    ) -> list[str]:
        """
        Apply chat templates and generate responses.

        Parameters
        ----------
        messages : list
            List of messages for input to model.

        Returns
        -------
        responses : list
            List of generated responses.
        """
        responses = []

        batched_inputs = [
            messages[i : i + self.config["batch_size"]]
            for i in range(0, len(messages), self.config["batch_size"])
        ]

        for batch in tqdm(batched_inputs, desc="Generating batch responses"):

            inputs = self.tokenizer.apply_chat_template(
                batch,
                padding=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=self.config["temperature"],
            )

            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return responses

    def parse_outputs(
            self,
            inputs: list[str],
            outputs: list[str],
    ) -> list:
        """
        Parse outputs from model.

        Parameters
        ----------
        outputs : list
            List of generated responses.

        Returns
        -------
        responses : list
            List of parsed responses.
        """
        responses = []

        for in_out_pair in tqdm(zip(inputs, outputs), desc="Parsing outputs"):
            responses.append(in_out_pair[1].split(in_out_pair[0])[-1].strip())

        return responses

    def run(
            self,
            save: bool = True,
    ) -> dict:
        """
        Run the model on the dataset.
        """
        inputs = self.load_inputs()
        messages = self.assemble_messages(inputs)
        outputs = self.apply_and_generate(messages)
        responses = self.parse_outputs(inputs, outputs)
        if save:
            self.save_results(inputs, responses)

        return list(zip(inputs, responses))


class GoogleGemma(GenerationBase):

    def __init__(
            self,
            config: dict,
    ) -> None:

        super().__init__(config)

        # check model is a Mistral model
        if config['model'].split('/')[0] != 'google':
            raise ValueError("Model(/family) doesn't appear correct.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"],
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model"],
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.device = self.model.device
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def assemble_messages(
            self,
            inputs: list[str],
    ) -> list:
        """
        Assemble messages for input to model.

        Parameters
        ----------
        inputs : list
            List of input questions.

        Returns
        -------
        messages : list
            List of messages for input to model.
        """
        messages = []

        for i in tqdm(inputs, desc="Assembling messages"):
            messages.append(
                f'{self.config["system_content"]} {i}'
            )

        return messages

    def apply_and_generate(
            self,
            messages: list[str],
    ) -> list[str]:
        """
        Apply chat templates and generate responses.

        Parameters
        ----------
        messages : list
            List of messages for input to model.

        Returns
        -------
        responses : list
            List of generated responses.
        """
        responses = []

        batched_inputs = [
            messages[i : i + self.config["batch_size"]]
            for i in range(0, len(messages), self.config["batch_size"])
        ]

        for batch in tqdm(batched_inputs, desc="Generating batch responses"):

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt").to(self.device)['input_ids']

            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=self.config["temperature"],
            )

            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return responses

    def parse_outputs(
            self,
            inputs: list[str],
            outputs: list[str],
    ) -> list:
        """
        Parse outputs from model.

        Parameters
        ----------
        inputs : list
            List of input questions.
        outputs : list
            List of generated responses.

        Returns
        -------
        responses : list
            List of parsed responses.
        """
        responses = []

        for in_out_pair in tqdm(zip(inputs, outputs), desc="Parsing outputs"):
            responses.append(in_out_pair[1].split(in_out_pair[0])[-1].strip())

        return responses

    def run(
            self,
            save: bool = True,
    ) -> dict:
        """
        Run the model on the dataset.
        """
        inputs = self.load_inputs()
        messages = self.assemble_messages(inputs)
        outputs = self.apply_and_generate(messages)
        if save:
            self.save_results(inputs, outputs)

        return list(zip(inputs, outputs))

class OpenAI(GenerationBase):

    def __init__(
            self,
            config: dict,
    ) -> None:

        super().__init__(config)

        if config['model'][:3] != 'gpt':
            raise ValueError("Model(/family) doesn't appear correct.")

    def assemble_messages(
            self,
            inputs: list[str],
    ) -> list:
        """
        Assemble messages for input to model.

        Parameters
        ----------
        inputs : list
            List of input questions.

        Returns
        -------
        messages : list
            List of messages for input to model.
        """
        messages = []

        for i in tqdm(inputs, desc="Assembling messages"):
            messages.append(
                f'{self.config["system_content"]} {i}'
            )

        return messages

    def apply_and_generate(
            self,
            messages: list[str],
    ) -> list[str]:
        """
        Apply chat templates and generate responses.

        Parameters
        ----------
        messages : list
            List of messages for input to model.

        Returns
        -------
        responses : list
            List of generated responses.
        """
        responses = []

        for message in tqdm(messages, desc="Generating responses"):
            response = OpenAI.completion.create(
                engine=self.config["model"],
                prompt=message,
                max_tokens=512,
                temperature=self.config["temperature"],
            )
            responses.append(response.choices[0].text.strip())

        return responses









if __name__ == "__main__":


    MODELS = {
        'meta-llama/Meta-Llama-3-8B-Instruct': MetaLlama,
        'meta-llama/Meta-Llama-3.1-8B-Instruct': MetaLlama,
        'mistralai/Mistral-7B-Instruct-v0.3': Mistral,
        'microsoft/Phi-3-mini-128k-instruct': MicrosoftPhi,
        'microsoft/Phi-3-small-128k-instruct': MicrosoftPhi,
        'microsoft/Phi-3.5-mini-instruct': MicrosoftPhi,
        'google/gemma-7b': GoogleGemma,
        'google/gemma-2-9b-it': GoogleGemma,
    }

    parser = argparse.ArgumentParser(description='Run model on dataset.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--description', type=str, default='', help='Description of run.')
    parser.add_argument('--examples', type=int, default=16, help='Number of examples to run.')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Model to run.')
    parser.add_argument('--save', type=bool, default=False, help='Save results.')
    parser.add_argument('--source', type=str, default='/app/resources/data/strategyqa/dev.jsonl', help='Source data.')
    parser.add_argument('--system_content', type=str, default='Answer the following question.', help='System content.')
    parser.add_argument('--temperature', type=float, default=0.2, help='Generation temperature.')
    args = parser.parse_args()

    config = {
        "batch_size": args.batch_size,
        "description": args.description,
        "examples": args.examples,
        "model": args.model,
        "source": args.source,
        "system_content": args.system_content,
        "temperature": args.temperature,
    }

    print(f"Running with config:\n{json.dumps(config, indent=4)}")
    model = MODELS[args.model](config)
    model.run(save=args.save)
    print(f"Saved to {model.filename}.")
