import argparse
import datetime
import json
import logging
import random
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)

STEP_NUM = {
    "A2": 6,
    "B2": 12,
    "C2": 8,
    "D2": 8,
}


class LLMParticipant:

    def __init__(self, config: dict) -> None:

        self.config = config
        self.MODEL = config["MODEL"]
        self.model_family = self.MODEL.split("/")[0]
        self.TEMPERATURE = config["TEMPERATURE"]
        self.BATCH_SIZE = config["BATCH_SIZE"]
        self.EXAMPLE = config["EXAMPLE"]
        self.shot = config["SHOT"]
        self.INPUT = config["INPUT"]
        self.SYSTEM_CONTENT = config["SYSTEM_CONTENT"]
        self.FORMAT_VERSION = config["FORMAT_VERSION"]

        self.data = self.load_input_questions()

        print(f"Loading model {self.MODEL}... ", end="")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL,
            # token='hf_xxx'
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # local_files_only=True
        )
        print("done.")

        # if self.model_family not in ['google']:
        #     self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #     self.model.to(self.device)

        self.device = self.model.device

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def load_input_questions(
        self,
    ) -> dict:

        with Path(self.INPUT).open(encoding="utf-8") as f:
            data = json.load(f)

        return data

    def assemble_messages(
        self,
        data: dict,
    ) -> list[list[dict]]:

        inputs = []
        for example in data.values():

            user_content = example["question"]

            if self.model_family in [""]:

                messages = [
                    {"role": "system", "content": self.SYSTEM_CONTENT},
                    {"role": "user", "content": user_content},
                ]

            elif self.model_family in ["mistralai", "microsoft"]:
                messages = [
                    {
                        "role": "user",
                        "content": self.SYSTEM_CONTENT + " " + user_content,
                    },
                ]

            elif self.model_family in ["google"]:
                messages = f"{self.SYSTEM_CONTENT} {user_content}"

            inputs.append(messages)

        return inputs

    def apply_template_generate_response(self, inputs) -> list[str]:

        responses = []
        batched_inputs = [
            inputs[i : i + self.BATCH_SIZE]
            for i in range(0, len(inputs), self.BATCH_SIZE)
        ]

        for batch in batched_inputs:

            if self.model_family in ["", "mistralai", "microsoft"]:
                input_messages = self.tokenizer.apply_chat_template(
                    batch,
                    padding=True,
                    return_tensors="pt",
                    tokenize=True,
                    add_generation_prompt=True,
                ).to(self.device)
            elif self.model_family in ["google"]:
                input_messages = self.tokenizer(
                    batch, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)["input_ids"]
            outputs = self.model.generate(
                input_messages,
                max_new_tokens=256,
                do_sample=True,
                temperature=self.TEMPERATURE,
            )
            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return responses

    def save_results(self, results: list[dict]):

        str_shot_dict = {0: "zero", 1: "one", 3: "few"}
        output_filename = datetime.datetime.isoformat(datetime.datetime.now())
        output_path = Path(
            "participant",
            "outputs",
            f"{Path(self.INPUT).stem}",
            f"{str_shot_dict[self.shot]}",
            f"{output_filename}.json",
        )
        log = Path("participant/log.json")
        print(f"Saving results in {output_path}... ", end="")

        outputs = {"config": self.config, "results": results}

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4)
        print("done.")

        with log.open("r", encoding="utf-8") as f:
            log_data = json.load(f)

        log_data[output_filename] = self.config
        # sort log
        log_data = dict(sorted(log_data.items()))

        with log.open("w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)

    def participate(self):

        print(
            f"Assembling messages for {len(self.data)} inputs... ", end="", flush=True
        )
        inputs = self.assemble_messages(self.data)
        print("done.", flush=True)
        print(f"Generating responses for {len(inputs)} inputs... ", end="", flush=True)
        responses = self.apply_template_generate_response(inputs)
        print("done.", flush=True)
        outputs = []
        for q, r in zip(self.data.keys(), responses):
            t = self.data[q]["template_id"]
            outputs.append(
                {
                    "question_id": q,
                    "template_id": t,
                    "response": r,
                }
            )

        self.save_results(outputs)

        return outputs


class AnalyseLLMParticipant:
    def __init__(self, output_file):
        self.output_file = output_file
        self.config, self.results = self.load_data()
        self.questions = self.load_questions()
        self.df = self.split_by_prompt()
        self.parse_generation_for_selection()
        self.response_eq_example()
        self.response_length_gt_steps()
        self.response_eq_all_steps()
        self.proportion_of_all_steps()
        self.meta_object_selection()

    def load_data(self):
        with open(self.output_file) as f:
            data = json.load(f)

        config = data["config"]
        results = data["results"]

        return config, results

    def load_questions(self):

        questions = Path(f"../resources/data/select.json")
        with open(questions) as f:
            data = json.load(f)

        return data

    def split_by_prompt(self):

        qids = [r["question_id"] for r in self.results]
        templates = [r["template_id"] for r in self.results]
        responses = [r["response"] for r in self.results]

        model_family = self.config["MODEL"].split("/")[0]
        system = []
        user = []
        generation = []

        if model_family in [""]:

            system_prompt = "system\n\n"
            user_prompt = "user\n\n"
            generation_prompt = "assistant\n\n"

            for response in responses:
                system.append(
                    response[len(system_prompt) : response.index(user_prompt)]
                )
                user.append(
                    response[
                        response.index(user_prompt)
                        + len(user_prompt) : response.index(generation_prompt)
                    ]
                )
                generation.append(
                    response[
                        response.index(generation_prompt) + len(generation_prompt) :
                    ]
                )

        elif model_family in ["mistralai"]:

            generation_prompt = "   "
            for response in responses:
                system.append(None)
                user.append(response[: response.index(generation_prompt)])
                generation.append(
                    response[
                        response.index(generation_prompt) + len(generation_prompt) :
                    ]
                )

        elif model_family in ["google", "microsoft"]:

            # generation simply follows system prompt, so split after system prompt
            system_prompt = self.config["SYSTEM_CONTENT"]
            # get index at which system prompt ends
            system_prompt_end = len(system_prompt)
            for response in responses:
                system.append(None)
                user.append(response[:system_prompt_end])
                generation.append(response[system_prompt_end:])

        # dictionary with {qid: [system, user, generation]}
        return pd.DataFrame(
            {
                "qid": qids,
                "template": templates,
                "num_steps": [STEP_NUM[t] for t in templates],
                "system": system,
                "user": user,
                "generation": generation,
            }
        )

    def get_template_from_qid(
        self,
        qid,
        return_steps_or_id: str = "steps",
    ):

        template = self.questions[qid]["template_id"]

        if return_steps_or_id == "steps":
            return STEP_NUM[template]
        else:
            return template

    def parse_generation_for_selection(self):

        def matches_format(generation):
            # regex for matching lists e.g., [1, 2, 3] or [a, b, c]
            r = re.compile(r"\[.*?\]")
            matches = r.findall(generation)
            if matches:
                match = matches[-1]
                try:
                    test = [str(m) for m in match[1:-1].split(",")]
                except ValueError:
                    test = None
                return match

        def valid_generation(template, generation):
            if generation is not None:
                try:
                    valid = [int(m) for m in generation[1:-1].split(",")]
                except ValueError:
                    valid = None
            else:
                valid = None

            # compare to number in different column
            if valid is not None and len(valid) <= STEP_NUM[template]:
                return valid

            return None

        self.df["matches_format"] = self.df["generation"].apply(
            lambda x: matches_format(x)
        )
        # compare to number in df['num_steps']
        self.df["valid_generation"] = self.df[["template", "matches_format"]].apply(
            lambda x: valid_generation(x["template"], x["matches_format"]), axis=1
        )

    def response_eq_example(self):

        example = self.config["EXAMPLE"]
        # count number of times example == df['matches_format']
        self.df["response_eq_example"] = self.df["matches_format"].apply(
            lambda x: True if x == example else False
        )

    def response_length_gt_steps(self):

        # count cases where the assistant generation has more steps than the example
        self.df["response_length_gt_steps"] = self.df.apply(
            lambda row: (
                True
                if row["valid_generation"] is not None
                and len(row["valid_generation"]) > row["num_steps"]
                else False
            ),
            axis=1,
        )

    def response_eq_all_steps(self):

        # count cases where assistant generation is just the full list of steps
        self.df["response_eq_all_steps"] = self.df.apply(
            lambda row: (
                True
                if row["valid_generation"]
                == [i for i in range(1, row["num_steps"] + 1)]
                else False
            ),
            axis=1,
        )

    def proportion_of_all_steps(self):

        # calculate the proportion of steps that the model selected
        def proportion_of_all_steps(generation, template):
            if generation is not None:
                return len(generation) / STEP_NUM[template]
            return None

        self.df["proportion_of_all_steps"] = self.df.apply(
            lambda row: proportion_of_all_steps(
                row["valid_generation"], row["template"]
            ),
            axis=1,
        )

    def meta_object_selection(self):

        example = self.questions[list(self.questions.keys())[0]]
        xp = example["explanation"]
        meta_steps = [i["step"] for i in xp if i["label"] == "meta"]
        object_steps = [i["step"] for i in xp if i["label"] == "object"]

        def meta_selections(generation):
            if generation is not None:
                return [i for i in generation if i in meta_steps]
            return None

        def meta_selections_pc(generation):
            if generation is not None:
                return len([i for i in generation if i in meta_steps]) / len(meta_steps)
            return None

        def object_selections(generation):
            if generation is not None:
                return [i for i in generation if i in object_steps]
            return None

        def object_selections_pc(generation):
            if generation is not None:
                return len([i for i in generation if i in object_steps]) / len(
                    object_steps
                )
            return None

        self.df["meta_selections"] = self.df["valid_generation"].apply(
            lambda x: meta_selections(x)
        )
        self.df["meta_selections_pc"] = self.df["valid_generation"].apply(
            lambda x: meta_selections_pc(x)
        )
        self.df["object_selections"] = self.df["valid_generation"].apply(
            lambda x: object_selections(x)
        )
        self.df["object_selections_pc"] = self.df["valid_generation"].apply(
            lambda x: object_selections_pc(x)
        )


class AnalyseLLMParticipants:

    def __init__(self, shots, output_file):
        self.shots = shots
        self.output_file = output_file
        self.config, self.results = self.load_data()
        self.questions = self.load_questions()

    def load_questions(self):

        questions = Path(f"../resources/data/select.json")
        with open(questions) as f:
            data = json.load(f)

        return data

    def load_data(self):
        with open(self.output_file) as f:
            data = json.load(f)

        config = data["config"]
        results = data["results"]

        return config, results


if __name__ == "__main__":

    MODEL = "microsoft/Phi-3-small-128k-instruct"
    TEMPERATURE = 0.3
    BATCH_SIZE = 20
    EXAMPLE = "[1, 2, 3, 4]"
    FORMAT_VERSION = 2
    SYSTEM_CONTENT = f"Your role is to select, from a list the steps, only those that are most important for inclusion in a summary explanation of that process. Format your output as a list, for example {EXAMPLE}. Output only this short summary paragraph and nothing else.".format(
        EXAMPLE
    )
    INPUTS = "../resources/data/select.json"
    DESCRIPTION = "testing"

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, default="queue.json")
    parser.add_argument("--model", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--example", type=str)
    parser.add_argument("--system_content", type=str)
    parser.add_argument("--inputs", type=str)
    parser.add_argument("--description", type=str)
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
            "DESCRIPTION": DESCRIPTION,
        }

    else:
        with Path(args.batch).open(encoding="utf-8") as f:
            queue = json.load(f)

        print(f"Processing {len(queue)} configs...")
        for i, config in enumerate(queue.items(), 1):
            description, config = config
            print(f"({i}/{len(queue)}) {description}...")
            participant = LLMParticipant(config)
            participant.participate()
            print(f"Completed {description}.")

            with Path(args.batch).open("r", encoding="utf-8") as f:
                partial_queue = json.load(f)

            # remove key config[0] from queue
            del partial_queue[description]

            with Path(args.batch).open("w", encoding="utf-8") as f:
                json.dump(partial_queue, f, indent=4)
