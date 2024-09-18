"""Module containing base classes for generating responses using LLMs."""

import argparse
import datetime
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
import torch
from data import Dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('datasets').setLevel(logging.ERROR)

os.environ['TRANSFORMERS_NO_TQDM'] = '1'

PWD = Path.cwd()


class GenerationConfig:
    """Class for configuration of generation tasks."""

    def __init__(
        self,
        batch_size: int = 8,
        description: str = '',
        examples: int = 16,
        filename: str = '',
        model: str = 'google/gemma-2-9b-it',
        save: bool | str = True,
        source: str = 'StrategyQA/dev.jsonl',
        system_content='Answer the following question.',
        temperature=0.2,
    ):
        """Initialize the GenerationConfig class.

        Parameters
        ----------
        batch_size: int
            Batch size for generation.
        description: str
            Description of the run.
        examples: int
            Number of examples to run.
        filename: str
            Filename to save the results. Overwritten if save=True. Use save='<filename>' to specify a filename.
        model: str
            Model to run.
        save: bool | str
            Whether to save the results.
        source: str
            Source data.
        system_content: str
            System content.
        temperature: float
            Generation temperature.

        """
        self.batch_size = batch_size
        self.description = description
        self.examples = examples
        if isinstance(save, bool) and save:
            self.filename = f'{datetime.datetime.now().replace(microsecond=0).isoformat()}.jsonl'
        elif isinstance(save, str):
            self.filename = save
        self.model = model
        self.save = save
        self.source = source
        self.system_content = system_content
        self.temperature = temperature

    def load_from_dict(
        self,
        config: dict,
    ) -> None:
        """Load configuration from dictionary.

        Parameters
        ----------
        config: dict
            Configuration dictionary.

        """
        for key, value in config.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
            else:
                raise ValueError(f'Invalid key in config: {key}')

    def __str__(self):
        """Return the string representation of the class."""
        return json.dumps(self.__dict__, indent=4)


class GenerationBase:
    """Base class for applying LLMs to datasets."""

    def __init__(
        self,
        config: GenerationConfig,
    ) -> None:
        """Initialize the GenerationBase class.

        Parameters
        ----------
        config: GenerationConfig
            Configuration containing model and data parameters.

        """
        self.config = config

    def load_inputs(
        self,
    ) -> pd.DataFrame:
        """Load inputs from source file as given in config.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing input questions.

        """
        dataset_path = Path(self.config.source)
        dataset_name = dataset_path.parent.name
        source = dataset_path.name
        dataset = Dataset(dataset_name, source)

        return dataset.load_data()

    def update_log(
        self,
        log_path: str | Path = Path(PWD, 'mini-project', 'outputs', 'log.jsonl'),
    ) -> pd.DataFrame:
        """Update log with new entry.

        Parameters
        ----------
        log_path: str | Path
            Path to log file.

        Returns
        -------
        new_log_entry_df: pd.DataFrame
            DataFrame with new log entry.

        """
        if not Path(log_path).exists():
            log = pd.DataFrame(columns=sorted(GenerationConfig().__dict__.keys()))
        else:
            log = pd.read_json(log_path, orient='records', lines=True)

        new_log_entry_df = pd.DataFrame([self.config.__dict__])
        log = pd.concat([log, new_log_entry_df], ignore_index=True)
        log.to_json(log_path, orient='records', lines=True)

        return new_log_entry_df

    def clean_math(
        self,
        response: str,
    ) -> str:
        """Clean up LaTeX math in response.

        Parameters
        ----------
        response: str
            Response containing LaTeX math.

        Returns
        -------
        response: str
            Response with LaTeX math cleaned up.

        """
        response = re.sub(r'\\text\{([^}]*)\}', r'\1', response)  # Replace \text{...} with ...
        response = re.sub(r'\\boxed\{([^}]*)\}', r'\1', response)  # Replace \boxed{...} with ...
        response = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', response)  # Replace \frac{a}{b} with a/b
        response = re.sub(r'\\times', 'x', response)  # Replace \times with x
        response = re.sub(r'\\\[|\\\]', '', response)  # Remove \[ and \]
        response = re.sub(r'\\\(|\\\)', '', response)  # Remove \( and \)

        return response

    def save_results(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Save results to output file.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing results.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing results.

        """
        output_dir = Path(PWD, 'mini-project', 'outputs')
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_json(output_dir / self.config.filename, orient='records', lines=True)
        self.update_log()

        return df

    def assemble_messages(
        self,
        df: pd.DataFrame,
    ) -> NotImplementedError:
        """Assemble messages for input to model.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing input questions.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with assembled messages.

        """
        raise NotImplementedError('This method should be overridden by subclasses')

    def apply_and_generate(
        self,
        df: pd.DataFrame,
    ) -> NotImplementedError:
        """Apply chat templates and generate responses.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing messages for input to model.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with generated responses.

        """
        raise NotImplementedError('This method should be overridden by subclasses')

    def parse_responses(
        self,
        df: pd.DataFrame,
    ) -> NotImplementedError:
        """Parse outputs from model.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing generated responses.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with parsed responses.

        """
        raise NotImplementedError('This method should be overridden by subclasses')

    def run(
        self,
    ) -> pd.DataFrame:
        """Run the model on the dataset.

        Parameters
        ----------
        save: bool | str
            Whether to save the results.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing results.

        """
        df = self.load_inputs()
        df = df.sample(n=self.config.examples, random_state=72) if self.config.examples > 0 else df
        df = self.assemble_messages(df)
        df = self.apply_and_generate(df)
        df = self.parse_responses(df)
        if self.config.save:
            self.save_results(df)

        print(df.head())

        return df


class MetaLlama(GenerationBase):
    """Implementation of Llama chat templates etc for application to dataset.

    This class handles the generation of chat templates and their application to datasets.
    """

    def __init__(
        self,
        config: dict,
    ) -> None:
        """Initialize the MetaLlama instance.

        Parameters
        ----------
        config: dict
            Configuration dictionary containing model and data parameters.

        """
        super().__init__(config)

        assert config.model.split('/')[0] == 'meta-llama', 'Model must be a Meta-Llama model.'

        local_path = '/nfs/public/hf/models/'
        full_model_path = local_path + config.model

        self.tokenizer = self.tokenizer = AutoTokenizer.from_pretrained(
            full_model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            full_model_path,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.device = self.model.device
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

    def assemble_messages(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Assemble messages for input to model.

        Parameters
        ----------
        df: list
            List of input questions.

        Returns
        -------
        messages: list
            List of messages for input to model.

        """
        messages = []

        for i in df['question']:
            messages.append(
                [
                    {'role': 'system', 'content': self.config.system_content},
                    {'role': 'user', 'content': i},
                ]
            )

        df['messages'] = messages

        return df

    def apply_and_generate(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """Apply chat templates and generate responses.

        Parameters
        ----------
        df: list
            List of messages for input to model.

        Returns
        -------
        responses: list
            List of generated responses.

        """
        responses = []
        messages = df['messages'].tolist()

        batched_inputs = [messages[i : i + self.config.batch_size] for i in range(0, len(messages), self.config.batch_size)]

        for batch in tqdm(batched_inputs, desc='Generating responses'):
            inputs = self.tokenizer.apply_chat_template(
                batch,
                padding=True,
                return_tensors='pt',
                add_generation_prompt=True,
            ).to(self.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=self.config.temperature,
            )

            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        df['response'] = responses

        return df

    def parse_responses(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Parse outputs from model.

        Parameters
        ----------
        df: list
            List of generated responses.

        Returns
        -------
        list
            List of parsed responses.

        """
        generation_prompt = 'assistant\n\n'

        responses = []

        for response in df['response']:
            responses.append(response.split(generation_prompt)[-1].strip())

        # clean up math
        responses = [self.clean_math(r) for r in responses]

        df['parsed_response'] = responses

        return df


class Mistral(GenerationBase):
    """Implementation of Mistral chat templates etc for application to dataset."""

    def __init__(
        self,
        config: dict,
    ) -> None:
        """Initialize the Mistral instance.

        Parameters
        ----------
        config: dict
            Configuration dictionary containing model and data parameters.

        """
        super().__init__(config)

        # check model is a Mistral model
        if config.model.split('/')[0] != 'mistralai':
            raise ValueError("Model(/family) doesn't appear correct.")

        self.tokenizer = self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.device = self.model.device
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

    def assemble_messages(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Assemble messages for input to model.

        Parameters
        ----------
        df: list
            List of input questions.

        Returns
        -------
        messages: list
            List of messages for input to model.

        """
        messages = []

        for i in df['question']:
            messages.append(
                [
                    {
                        'role': 'user',
                        'content': self.config.system_content + ' ' + i,
                    },
                ]
            )

        return messages

    def apply_and_generate(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """Apply chat templates and generate responses.

        Parameters
        ----------
        df: list
            List of messages for input to model.

        Returns
        -------
        responses: list
            List of generated responses.

        """
        responses = []
        messages = df['messages'].tolist()

        batched_inputs = [messages[i : i + self.config.batch_size] for i in range(0, len(messages), self.config.batch_size)]

        for batch in tqdm(batched_inputs, desc='Generating responses'):
            inputs = self.tokenizer.apply_chat_template(
                batch,
                padding=True,
                return_tensors='pt',
                add_generation_prompt=True,
            ).to(self.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=self.config.temperature,
            )

            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        df['response'] = responses

        return df

    def parse_responses(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Parse outputs from model.

        Parameters
        ----------
        df: list
            List of generated responses.

        Returns
        -------
        responses: list
            List of parsed responses.

        """
        responses = []
        inputs = df['question'].tolist()
        outputs = df['response'].tolist()

        for in_out_pair in zip(inputs, outputs):
            responses.append(in_out_pair[1].split(in_out_pair[0])[-1].strip())

        # clean up math
        responses = [self.clean_math(r) for r in responses]

        df['parsed_response'] = responses

        return df


class MicrosoftPhi(GenerationBase):
    """Implementation of Microsoft chat templates etc for application to dataset."""

    def __init__(
        self,
        config: dict,
    ) -> None:
        """Initialize the MicrosoftPhi instance.

        Parameters
        ----------
        config: dict
            Configuration dictionary containing model and data parameters.

        """
        super().__init__(config)

        # check model is a Mistral model
        if config.model.split('/')[0] != 'microsoft':
            raise ValueError("Model(/family) doesn't appear correct.")

        self.tokenizer = self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            attn_implementation='flash_attention_2',
            device_map='cuda',
            torch_dtype='auto',
            trust_remote_code=True,
        )

        self.device = self.model.device
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

    def assemble_messages(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Assemble messages for input to model.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing input questions.

        Returns
        -------
        messages: list
            List of messages for input to model.

        """
        messages = []

        for i in df['question']:
            messages.append(
                [
                    {
                        'role': 'user',
                        'content': self.config.system_content + ' ' + i,
                    },
                ]
            )

        df['messages'] = messages

        return df

    def apply_and_generate(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """Apply chat templates and generate responses.

        Parameters
        ----------
        df: list
            List of messages for input to model.

        Returns
        -------
        responses: list
            List of generated responses.

        """
        responses = []
        messages = df['messages'].tolist()

        batched_inputs = [messages[i : i + self.config.batch_size] for i in range(0, len(messages), self.config.batch_size)]

        for batch in tqdm(batched_inputs, desc='Generating responses'):
            inputs = self.tokenizer.apply_chat_template(
                batch,
                padding=True,
                return_tensors='pt',
                add_generation_prompt=True,
            ).to(self.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=self.config.temperature,
            )

            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        df['response'] = responses

        return df

    def parse_responses(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Parse outputs from model.

        Parameters
        ----------
        df: list
            List of generated responses.

        Returns
        -------
        responses: list
            List of parsed responses.

        """
        responses = []

        for q, r in zip(df['question'], df['response']):
            prefix = f'{self.config.system_content} {q}'
            # strip prefix from response
            responses.append(r[len(prefix) :].strip())

        # clean up math
        responses = [self.clean_math(r) for r in responses]

        df['parsed_response'] = responses

        return df


class GoogleGemma(GenerationBase):
    """Implementation of Google chat templates etc for application to dataset."""

    def __init__(
        self,
        config: dict,
    ) -> None:
        """Initialize the GoogleGemma instance.

        Parameters
        ----------
        config: dict
            Configuration dictionary containing model and data parameters.

        """
        super().__init__(config)

        # check model is a Mistral model
        if config.model.split('/')[0] != 'google':
            raise ValueError("Model(/family) doesn't appear correct.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.device = self.model.device
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

    def assemble_messages(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Assemble messages for input to model.

        Parameters
        ----------
        df: list
            List of input questions.

        Returns
        -------
        messages: list
            List of messages for input to model.

        """
        messages = []

        for i in df['question']:
            messages.append(f'{self.config.system_content} {i}')

        df['messages'] = messages

        return df

    def apply_and_generate(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """Apply chat templates and generate responses.

        Parameters
        ----------
        df: list
            List of messages for input to model.

        Returns
        -------
        responses: list
            List of generated responses.

        """
        responses = []
        messages = df['messages'].tolist()

        batched_inputs = [messages[i : i + self.config.batch_size] for i in range(0, len(messages), self.config.batch_size)]

        for batch in tqdm(batched_inputs, desc='Generating responses'):
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt',
            ).to(self.device)['input_ids']

            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=self.config.temperature,
            )

            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        df['response'] = responses

        return df

    def parse_responses(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Parse outputs from model.

        Parameters
        ----------
        df: list
            List of generated responses

        Returns
        -------
        responses: list
            List of parsed responses.

        """
        responses = []

        for q, r in zip(df['question'], df['response']):
            prefix = f'{self.config.system_content} {q}'
            # strip prefix from response
            responses.append(r[len(prefix) :].strip())

        # clean up math
        responses = [self.clean_math(r) for r in responses]

        df['parsed_response'] = responses

        return df


class OpenAIGPT4omini(GenerationBase):
    """Placeholder for OpenAI GPT-4 mini implementation."""

    def __init__(
        self,
        config: dict,
    ) -> None:
        """Initialize the OpenAIGPT4mini instance.

        Parameters
        ----------
        config: dict
            Configuration dictionary containing model and data parameters.

        """
        super().__init__(config)

        # check model is an OpenAI model
        self.provider, self.model = config.model.split('/')
        if self.provider != 'openai':
            raise ValueError("Model(/family) doesn't appear correct.")

    def assemble_messages(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Assemble messages for input to model.

        Parameters
        ----------
        df: list
            List of input questions.

        Returns
        -------
        messages: list
            List of messages for input to model.

        """
        messages = []

        for q in df['question']:
            messages.append([{'role': 'system', 'content': self.config.system_content}, {'role': 'user', 'content': q}])

        df['messages'] = messages

        return df

    def apply_and_generate(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """Apply chat templates and generate responses.

        Parameters
        ----------
        df: list
            List of messages for input to model.

        Returns
        -------
        responses: list
            List of generated responses.

        """
        responses = []
        messages = df['messages'].tolist()
        client = OpenAI()

        for message in messages:
            response = client.chat.completions.create(
                messages=message,
                model=self.model,
                max_tokens=512,
                temperature=self.config.temperature,
            )

            responses.append(response.choices[0].message.content)

        df['response'] = responses

        return df

    def parse_responses(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Parse outputs from model.

        Parameters
        ----------
        df: list
            List of generated responses.

        Returns
        -------
        responses: list
            List of parsed responses.

        """
        df['parsed_response'] = df['response']

        # clean up math
        df['parsed_response'] = [self.clean_math(r) for r in df['parsed_response']]

        return df


if __name__ == '__main__':

    def batch(
        filepath: str,
    ) -> None:
        """Load config from file and run model on dataset.

        Parameters
        ----------
        filepath: str
            Path to config file.

        """
        cfgs = pd.read_json(filepath, orient='records', lines=True)
        cfgs = cfgs.to_dict(orient='records')

        for cfg in tqdm(cfgs, desc='Running batch'):
            # tqdm.write(f'Running with config={cfg}')
            config = GenerationConfig(**cfg)
            model = MODELS[config.model](config)
            model.run()

    SOURCES = [
        'Franklin/full_study.jsonl',
        'GSM8k/dev.jsonl',
        'HotpotQA/hotpot_test_fullwiki_v1.jsonl',
        'StrategyQA/dev_decomp_3_plus.jsonl',
    ]

    MODELS = {
        'google/gemma-7b': GoogleGemma,
        'google/gemma-2-9b-it': GoogleGemma,
        'meta-llama/Meta-Llama-3-8B-Instruct': MetaLlama,
        'meta-llama/Meta-Llama-3.1-8B-Instruct': MetaLlama,
        'mistralai/Mistral-7B-Instruct-v0.3': Mistral,
        'microsoft/Phi-3-mini-128k-instruct': MicrosoftPhi,
        'microsoft/Phi-3-small-128k-instruct': MicrosoftPhi,
        'microsoft/Phi-3.5-mini-instruct': MicrosoftPhi,
        'openai/gpt-4o-mini': OpenAIGPT4omini,
    }

    parser = argparse.ArgumentParser(description='Run model on dataset.')
    parser.add_argument('--batch', type=str, default='', help='Batch file.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--description', type=str, default='', help='Description of run.')
    parser.add_argument('--examples', type=int, default=16, help='Number of examples to run.')
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Meta-Llama-3.1-8B-Instruct',
        help='Model to run.',
        choices=MODELS.keys(),
    )
    parser.add_argument('--save', type=bool, default=False, help='Save results.')
    parser.add_argument(
        '--source',
        type=str,
        default='Franklin/full_study.jsonl',
        help='Source data.',
        choices=SOURCES,
    )
    parser.add_argument(
        '--system-content',
        type=str,
        default='Answer the following question.',
        help='System content.',
    )
    parser.add_argument('--temperature', type=float, default=0.2, help='Generation temperature.')
    args = parser.parse_args()

    if args.batch:
        batch(args.batch)

    else:
        config = GenerationConfig(
            batch_size=args.batch_size,
            description=args.description,
            examples=args.examples,
            model=args.model,
            save=args.save,
            source=args.source,
            system_content=args.system_content,
            temperature=args.temperature,
        )

        print(f'Running with config:\n{json.dumps(config.__dict__, indent=4)}')
        model = MODELS[args.model](config)
        model.run()
        if args.save:
            print(f'Saved to {model.config.filename}.')
