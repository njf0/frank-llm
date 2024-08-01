import datetime
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# FRANKLIN: Frank Library of Ideal Narratives (?!)

class Franklin:
    """
    Base class for applying LLMs to dataset of Frank-style questions.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model and data parameters.
    """
    def __init__(
        self,
        config: dict,
    ) -> None:

        self.config = config

    def load_inputs(
        self,
    ) -> list:
        """
        Load inputs from source file as given in config.
        """
        inputs = Path(self.config["source"])
        with inputs.open("r", encoding="utf-8") as f:
            inputs = [v["question"] for k, v in json.load(f).items()][:self.config["examples"]]

        return inputs

    def save_results(
        self,
        inputs: list,
        outputs: list,
    ) -> None:
        """
        Save results to output file.

        Parameters
        ----------
        inputs : list
            List of input questions.
        outputs : list
            List of generated responses.
        """
        if not Path('/app/plan/outputs').exists():
            Path('/app/plan/outputs').mkdir()
        logfile = Path('/app/plan/outputs/log.csv')
        if not logfile.exists():
            with open(logfile, 'w', encoding='utf-8') as f:
                df = pd.DataFrame()
                df.to_csv(f, index=True)

        df = pd.DataFrame(
            {
                "input": inputs,
                "output": outputs,
            }
        )

        filename = datetime.datetime.isoformat(datetime.datetime.now())

        df.to_csv(f"/app/plan/outputs/{filename}.csv", index=False)

        with open('/app/plan/outputs/log.csv', 'r', encoding='utf-8') as f:
            log = pd.read_csv(f, index_col=0)

        df = pd.DataFrame(columns=self.config.keys(), data=[self.config.values()], index=[filename])
        log = pd.concat([log, df])

        with open('/app/plan/outputs/log.csv', 'w', encoding='utf-8') as f:
            log.to_csv(f, index=True)

        return df


class LlamaTest(Franklin):
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
    ) -> pd.DataFrame:
        """
        Run the model on the dataset.
        """
        inputs = self.load_inputs()
        messages = self.assemble_messages(inputs)
        outputs = self.apply_and_generate(messages)
        responses = self.parse_outputs(outputs)
        self.save_results(inputs, responses)

        return {"inputs": inputs, "outputs": responses}
