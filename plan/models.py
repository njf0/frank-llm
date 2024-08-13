import random

import torch
from base import GenerationBase
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# FRANKLIN: Frank Library of Ideal Narratives (?!)

class Llama(GenerationBase):
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

        assert config['model'].split('/')[0] == 'nfs', 'Model must be a Meta-Llama model.'

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
                    {"role": "system", "content": self.config["system_content"]},
                    {"role": "user", "content": i},
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
        generation_prompt = "assistant\n\n"

        responses = []

        for output in tqdm(outputs, desc="Parsing outputs"):
            responses.append(output.split(generation_prompt)[-1].strip())

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
        responses = self.parse_outputs(outputs)
        if save:
            self.save_results(inputs, responses)

        return list(zip(inputs, responses))

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

if __name__ == "__main__":

    config = {
        "batch_size": 16,
        "description": "Test addition of description column.",
        "examples": 16,
        # "model": "/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B-Instruct",
        'model': 'mistralai/Mistral-7B-Instruct-v0.3',
        "source": "/app/resources/data/full_study.json",
        "system_content": "Answer the following question.",
        'temperature': 0.2
    }

    test = Mistral(config)
    results = test.run(save=False)
    print(*random.sample(results, 10), sep='\n')
