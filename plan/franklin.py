import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class Franklin:

    def __init__(
        self,
        config: dict,
    ) -> None:

        self.config = config

    def load_inputs(
        self,
    ) -> list:

        inputs = Path(self.config["source"])
        with inputs.open("r") as f:
            inputs = [v["question"] for k, v in json.load(f).items()]

        return inputs

    def save_results(
        self,
        inputs: list,
        outputs: list,
    ) -> None:

        df = pd.DataFrame(
            {
                "input": inputs,
                "output": outputs,
            }
        )

        return df


class LlamaTest(Franklin):

    def __init__(
        self,
        config: dict,
    ) -> None:

        super().__init__(config)

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
                max_new_tokens=256,
                do_sample=True,
                temperature=self.config["temperature"],
            )

            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return responses

    def parse_outputs(
        self,
        outputs: list[str],
    ) -> list:

        system_prompt = "system\n\n"
        user_prompt = "user\n\n"
        generation_prompt = "assistant\n\n"

        responses = []

        for output in outputs:
            responses.append(output.split(generation_prompt)[-1].strip())

        return responses

    def run(
        self,
    ) -> pd.DataFrame:

        inputs = self.load_inputs()
        messages = self.assemble_messages(inputs)
        outputs = self.apply_and_generate(messages)
        responses = self.parse_outputs(outputs)
        # df = self.save_results(inputs, responses)

        return {"inputs": inputs, "outputs": responses}
