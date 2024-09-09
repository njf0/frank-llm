"""Module containing base classes for generating responses using LLMs."""

import argparse
import datetime
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from data import Dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger('transformers').setLevel(logging.ERROR)

PWD = Path.cwd()

CONFIG_TYPES = {
    'batch_size': int,
    'examples': int,
    'description': str,
    'model': str,
    'temperature': float,
    'source': str,
    'system_content': str,
}


class GenerationBase:
    """Base class for applying LLMs to dataset of Frank-style questions.

    Parameters
    ----------
    config: dict
        Configuration dictionary containing model and data parameters.

    """

    def __init__(
        self,
        cfg: dict,
    ) -> None:
        """Initialize the GenerationBase class.

        Parameters
        ----------
        cfg: dict
            Configuration dictionary containing model and data parameters.

        """
        # type-check config against CONFIG_TYPES
        for key, value in cfg.items():
            if key not in CONFIG_TYPES:
                raise ValueError(f'Invalid key in config: {key}')
            if not isinstance(value, CONFIG_TYPES[key]):
                raise TypeError(f"Invalid type for '{key}' ({type(value)}). Should be {CONFIG_TYPES[key]}.")

        # make sure all keys are present
        if cfg.keys() != CONFIG_TYPES.keys():
            # report missing keys
            missing_keys = set(CONFIG_TYPES.keys()) - set(cfg.keys())
            raise ValueError(f"Missing keys in config: {', '.join(missing_keys)}")

        self.config = cfg
        self.filename = f'{datetime.datetime.now().replace(microsecond=0).isoformat()}.jsonl'

    def load_inputs(
        self,
    ) -> pd.DataFrame:
        """Load inputs from source file as given in config.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing input questions.

        """
        dataset_name = Path(self.config['source']).parent.name
        source = self.config['source']
        dataset = Dataset(dataset_name, source)

        return dataset()

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
        output_dir = Path(PWD, 'plan', 'outputs')
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_json(output_dir / self.filename, orient='records', lines=True)

        logfile = output_dir / 'log.jsonl'
        log = pd.read_json(logfile, orient='records', lines=True) if logfile.exists() else pd.DataFrame()

        self.config.update({'filename': self.filename})
        new_log_entry_df = pd.DataFrame([self.config])
        log = pd.concat([log, new_log_entry_df], ignore_index=True)
        log.to_json(logfile, orient='records', lines=True)

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

    def parse_outputs(
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
        save: bool = True,
    ) -> pd.DataFrame:
        """Run the model on the dataset.

        Parameters
        ----------
        save: bool
            Whether to save the results.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing results.

        """
        df = self.load_inputs()
        df = df.sample(n=self.config['examples'], random_state=72) if self.config['examples'] > 0 else df
        df = self.assemble_messages(df)
        df = self.apply_and_generate(df)
        df = self.parse_outputs(df)
        if save:
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

        assert config['model'].split('/')[0] == 'meta-llama', 'Model must be a Meta-Llama model.'

        local_path = '/nfs/public/hf/models/'
        full_model_path = local_path + config['model']

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

        for i in tqdm(df['question'], desc='Assembling messages'):
            messages.append(
                [
                    {'role': 'system', 'content': self.config['system_content']},
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

        batched_inputs = [
            messages[i : i + self.config['batch_size']] for i in range(0, len(messages), self.config['batch_size'])
        ]

        for batch in tqdm(batched_inputs, desc='Generating batch responses'):
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
                temperature=self.config['temperature'],
            )

            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        df['responses'] = responses

        return df

    def parse_outputs(
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

        for response in tqdm(df['responses'], desc='Parsing responses'):
            responses.append(response.split(generation_prompt)[-1].strip())

        df['parsed_responses'] = responses

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
        if config['model'].split('/')[0] != 'mistralai':
            raise ValueError("Model(/family) doesn't appear correct.")

        self.tokenizer = self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model'],
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model'],
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

        for i in tqdm(df['question'], desc='Assembling messages'):
            messages.append(
                [
                    {
                        'role': 'user',
                        'content': self.config['system_content'] + ' ' + i,
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

        batched_inputs = [
            messages[i : i + self.config['batch_size']] for i in range(0, len(messages), self.config['batch_size'])
        ]

        for batch in tqdm(batched_inputs, desc='Generating batch responses'):
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
                temperature=self.config['temperature'],
            )

            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        df['responses'] = responses

        return df

    def parse_outputs(
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
        outputs = df['responses'].tolist()

        for in_out_pair in tqdm(zip(inputs, outputs), desc='Parsing outputs'):
            # responses.append(output[output.index(generation_prompt) + len(generation_prompt):])
            responses.append(in_out_pair[1].split(in_out_pair[0])[-1].strip())

        df['parsed_responses'] = responses

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
        if config['model'].split('/')[0] != 'microsoft':
            raise ValueError("Model(/family) doesn't appear correct.")

        self.tokenizer = self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model'],
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model'],
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

        for i in tqdm(df['question'], desc='Assembling messages'):
            messages.append(
                [
                    {
                        'role': 'user',
                        'content': self.config['system_content'] + ' ' + i,
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

        batched_inputs = [
            messages[i : i + self.config['batch_size']] for i in range(0, len(messages), self.config['batch_size'])
        ]

        for batch in tqdm(batched_inputs, desc='Generating batch responses'):
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
                temperature=self.config['temperature'],
            )

            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        df['responses'] = responses

        return df

    def parse_outputs(
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

        for q, r in tqdm(zip(df['question'], df['responses']), desc='Parsing outputs'):
            prefix = f'{self.config["system_content"]} {q}'
            # strip prefix from response
            responses.append(r[len(prefix) :].strip())

        df['parsed_responses'] = responses

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
        if config['model'].split('/')[0] != 'google':
            raise ValueError("Model(/family) doesn't appear correct.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model'],
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model'],
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

        for i in tqdm(df['question'], desc='Assembling messages'):
            messages.append(f'{self.config["system_content"]} {i}')

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

        batched_inputs = [
            messages[i : i + self.config['batch_size']] for i in range(0, len(messages), self.config['batch_size'])
        ]

        for batch in tqdm(batched_inputs, desc='Generating batch responses'):
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
                temperature=self.config['temperature'],
            )

            responses += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        df['responses'] = responses

        return df

    def parse_outputs(
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

        for q, r in tqdm(zip(df['question'], df['responses']), desc='Parsing outputs'):
            prefix = f'{self.config["system_content"]} {q}'
            # strip prefix from response
            responses.append(r[len(prefix) :].strip())

        df['parsed_responses'] = responses

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
        self.provider, self.model = config['model'].split('/')
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

        for q in tqdm(df['question'], desc='Assembling messages'):
            messages.append([{'role': 'system', 'content': self.config['system_content']}, {'role': 'user', 'content': q}])

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

        for message in tqdm(messages, desc='Generating batch responses'):
            response = client.chat.completions.create(
                messages=message,
                model=self.model,
                max_tokens=512,
                temperature=self.config['temperature'],
            )

            responses.append(response.choices[0].message.content)

        df['responses'] = responses

        return df

    def parse_outputs(
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
        df['parsed_responses'] = df['responses']

        return df


if __name__ == '__main__':
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
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--description', type=str, default='', help='Description of run.')
    parser.add_argument('--examples', type=int, default=16, help='Number of examples to run.')
    parser.add_argument('--model', type=str, default='google/gemma-2-9b-it', help='Model to run.')
    parser.add_argument('--save', action='store_true', help='Save results.')
    parser.add_argument('--source', type=str, default='StrategyQA/dev.jsonl', help='Source data.')
    parser.add_argument(
        '--system_content',
        type=str,
        default='Answer the following question.',
        help='System content.',
    )
    parser.add_argument('--temperature', type=float, default=0.2, help='Generation temperature.')
    args = parser.parse_args()

    config = {
        'batch_size': args.batch_size,
        'description': args.description,
        'examples': int(args.examples),
        'model': args.model,
        'source': args.source,
        'system_content': args.system_content,
        'temperature': args.temperature,
    }

    print(f'Running with config:\n{json.dumps(config, indent=4)}')
    model = MODELS[args.model](config)
    model.run(save=args.save)
    print(f'Saved to {model.filename}.')
