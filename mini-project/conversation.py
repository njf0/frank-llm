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


class ConversationConfig:
    """Class for configuration of conversation tasks."""

    def __init__(
        self,
        batch_size: int = 8,
        description: str = '',
        examples: int = 16,
        filename: str = '',
        model: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        random_seed: int = 72,  # default for survey generation
        save: bool = False,
        source: str = 'StrategyQA/dev.jsonl',
        system_content: list | None = None,
        temperature: float = 0.2,
    ):
        """Initialize the ConversationConfig class.

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
        random_seed: int
            Random seed for generation.
        save: bool | str
            Whether to save the results.
        source: str
            Source data.
        system_content: list | None
            System content.
        temperature: float
            Generation temperature.

        """
        self.batch_size = batch_size
        self.description = description
        self.examples = examples
        if save:
            self.filename = f'{datetime.datetime.now().replace(microsecond=0).isoformat()}.jsonl'
        self.model = model
        self.random_seed = random_seed
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
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise ValueError(f'Invalid key in config: {key}')

    def __str__(self):
        """Return the string representation of the class."""
        return json.dumps(self.__dict__, indent=4)


class ConversationBase:
    """Base class for applying LLMs to datasets."""

    def __init__(
        self,
        config: ConversationConfig,
    ) -> None:
        """Initialize the ConversationBase instance.

        Parameters
        ----------
        config: ConversationConfig
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
        log_path: str | Path = Path(PWD, 'mini-project', 'studies', 'plan', 'output', 'log.jsonl'),
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
            log = pd.DataFrame(columns=sorted(ConversationConfig().__dict__.keys()))
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
        output_dir = Path(PWD, 'mini-project', 'studies', 'plan', 'output')
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_json(output_dir / self.config.filename, orient='records', lines=True)
        print(f'Outputs saved to {output_dir / self.config.filename}.')
        self.update_log()

        return df

    def assemble_conversations(
        self,
        df: pd.DataFrame,
    ) -> NotImplementedError:
        """Assemble conversations for input to model.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing input questions.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with assembled conversations.

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
            DataFrame containing conversations for input to model.

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
        df = df.sample(n=self.config.examples, random_state=self.config.random_seed) if self.config.examples > 0 else df
        df = self.assemble_conversations(df)
        df = self.apply_and_generate(df)
        df = self.parse_responses(df)
        if self.config.save:
            self.save_results(df)

        print(df.head())

        return df


class MetaLlama(ConversationBase):
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

        self.tokenizer = AutoTokenizer.from_pretrained(
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

    def assemble_conversations(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Assemble conversations for input to model.

        Parameters
        ----------
        df: list
            List of input questions.

        Returns
        -------
        conversations: list
            List of conversations for input to model.

        """
        conversations = []

        for i in df['question']:
            conversations.append(
                [
                    {'role': 'system', 'content': self.config.system_content[0]},
                    {'role': 'user', 'content': i},
                ]
            )

        df['conversations'] = conversations

        return df

    def apply_and_generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply chat templates and generate responses.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing conversations for input to model.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with generated responses.

        """
        full_conversations = []
        parsed_final_responses = []
        conversations = df['conversations'].tolist()
        # [
        #     {'role': 'system', 'content': config.system_content[0]},
        #     {'role': 'user', 'content': i},
        #     {'role': 'assistant', 'content': '...'},
        #     {'role': 'user', 'content': 'Now perform the steps in the plan you created.'},
        #     {'role': 'assistant', 'content': '...'},
        # ]
        subconfig = {k: v for k, v in self.config.__dict__.items() if k in ['model', 'source']}
        for conversation in tqdm(conversations, desc=str(subconfig)):
            history = conversation
            # print last part of history[0]
            cols = os.get_terminal_size().columns
            print(str(history[0])[:cols])
            print(str(history[1])[:cols])
            inputs = self.tokenizer.apply_chat_template(
                history,
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

            # parse initial response and add to history
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split('assistant')[-1].strip(' \n')
            prefix = ''
            parsed_response = response[len(prefix) :].strip(' \n')
            history.append({'role': 'assistant', 'content': parsed_response})
            print(str(history[-1])[:cols])

            # add a user message to prompt the assistant to perform the steps in the plan
            history.append({'role': 'user', 'content': self.config.system_content[1]})
            print(str(history[-1])[:cols])

            # generate the response
            inputs = self.tokenizer.apply_chat_template(
                history,
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

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split('assistant')[-1].strip(' \n')
            prefix = ''
            parsed_response = response[len(prefix) :].strip(' \n')
            history.append({'role': 'assistant', 'content': parsed_response})
            parsed_final_responses.append(parsed_response)
            print(str(history[-1])[:cols])

            full_conversations.append(history)

        df['conversation'] = full_conversations
        df['parsed_response'] = parsed_final_responses

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
        # def parse_response(row):
        #     object_response = row['final_response'].split('assistant')[-1].strip('\n')
        #     return object_response

        # # object_responses = df['response'].apply(lambda cell: cell[4]['content'].split('assistant')[-1].strip('\n'))

        # df['parsed_response'] = df.apply(lambda row: parse_response(row), axis=1)

        return df


class MicrosoftPhi(ConversationBase):
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            attn_implementation='flash_attention_2',
            device_map='cuda',
            torch_dtype='auto',
            trust_remote_code=True,
        )

        self.model.use_flash_attention = True
        self.device = self.model.device
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

    def assemble_conversations(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Assemble conversations for input to model.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing input questions.

        Returns
        -------
        conversations: list
            List of conversations for input to model.

        """
        conversations = []

        for i in df['question']:
            conversations.append(
                [
                    {'role': 'system', 'content': self.config.system_content[0]},
                    {'role': 'user', 'content': i},
                ]
            )

        df['conversation'] = conversations

        return df

    def apply_and_generate(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """Apply chat templates and generate responses.

        Parameters
        ----------
        df: list
            List of conversations for input to model.

        Returns
        -------
        responses: list
            List of generated responses.

        """
        full_conversations = []
        parsed_final_responses = []
        conversations = df['conversation'].tolist()
        # [
        #     {'role': 'system', 'content': config.system_content[0]},
        #     {'role': 'user', 'content': i},
        #     {'role': 'assistant', 'content': '...'},
        #     {'role': 'user', 'content': 'Now perform the steps in the plan you created.'},
        #     {'role': 'assistant', 'content': '...'},
        # ]
        subconfig = {k: v for k, v in self.config.__dict__.items() if k in ['model', 'source']}
        for conversation in tqdm(conversations, desc=str(subconfig)):
            history = conversation
            cols = os.get_terminal_size().columns
            print(str(history[0])[:cols])
            print(str(history[1])[:cols])

            inputs = self.tokenizer.apply_chat_template(
                history,
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

            # parse initial response and add to history
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split('assistant')[-1].strip(' \n')
            prefix = ' '.join([history[0]['content'], history[1]['content']])
            parsed_response = response[len(prefix) :].strip(' \n')
            history.append({'role': 'assistant', 'content': parsed_response})
            print(str(history[-1])[:cols])

            # add a user message to prompt the assistant to perform the steps in the plan
            history.append({'role': 'user', 'content': self.config.system_content[1]})
            print(str(history[-1])[:cols])

            # generate the response
            inputs = self.tokenizer.apply_chat_template(
                history,
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

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split('assistant')[-1].strip(' \n')
            prefix = ' '.join([history[0]['content'], history[1]['content'], history[2]['content'], history[3]['content']])
            parsed_response = response[len(prefix) :].strip(' \n')
            history.append({'role': 'assistant', 'content': parsed_response})
            parsed_final_responses.append(parsed_response)
            print(str(history[-1])[:cols])

            full_conversations.append(history)

        df['conversation'] = full_conversations
        df['parsed_response'] = parsed_final_responses

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
        # def parse_response(row):
        #     history = row['conversation']
        #     prefix = f"{history[0]['content']} {history[1]['content']} {history[2]['content']}"
        #     parsed_response = history[-1]['content'][len(prefix) :].strip(' \n')

        #     return parsed_response

        # df['parsed_response'] = df.apply(lambda row: parse_response(row), axis=1)

        return df


class GoogleGemma(ConversationBase):
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

    def assemble_conversations(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Assemble conversations for input to model.

        Parameters
        ----------
        df: list
            List of input questions.

        Returns
        -------
        conversations: list
            List of conversations for input to model.

        """
        conversations = []

        for i in df['question']:
            conversations.append([f'{self.config.system_content[0]} {self.config.system_content[1]} {i}'])

        df['conversations'] = conversations

        return df

    def apply_and_generate(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """Apply chat templates and generate responses.

        Parameters
        ----------
        df: list
            List of conversations for input to model.

        Returns
        -------
        responses: list
            List of generated responses.

        """
        full_conversations = []
        parsed_final_responses = []
        conversations = df['conversations'].tolist()

        # batched_inputs = [
        #     conversations[i : i + self.config.batch_size] for i in range(0, len(conversations), self.config.batch_size)
        # ]
        subconfig = {k: v for k, v in self.config.__dict__.items() if k in ['model', 'source']}
        for conversation in tqdm(conversations, desc=str(subconfig)):
            history = conversation[0]
            cols = os.get_terminal_size().columns
            print(history[:cols])

            inputs = self.tokenizer(
                history,
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

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            parsed_response = response[len(history) :].strip(' \n')
            print(repr(parsed_response)[:cols])  # print response including special characters
            full_conversations.append([history, parsed_response])
            parsed_final_responses.append(parsed_response)

        df['conversation'] = full_conversations
        df['parsed_response'] = parsed_final_responses

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
        # def parse_response(row):
        #     prefix = f'{" ".join(self.config.system_content)} {row["question"]}'
        #     return row['response'][len(prefix) :].strip(' \n')

        # # prefix = f'{" ".join(self.config.system_content)} {df["question"][df.index[0]]}'
        # df['parsed_response'] = df.apply(lambda row: parse_response(row), axis=1)

        return df


class OpenAIGPT4omini(ConversationBase):
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

    def assemble_conversations(
        self,
        df: pd.DataFrame,
    ) -> list:
        """Assemble conversations for input to model.

        Parameters
        ----------
        df: list
            List of input questions.

        Returns
        -------
        conversations: list
            List of conversations for input to model.

        """
        conversations = []

        for q in df['question']:
            conversations.append([{'role': 'system', 'content': self.config.system_content[0]}, {'role': 'user', 'content': q}])

        df['conversations'] = conversations

        return df

    def apply_and_generate(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply chat templates and generate responses.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame of conversations for input to model.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with generated responses.

        """
        full_conversations = []
        parsed_final_responses = []
        conversations = df['conversations'].tolist()
        client = OpenAI()

        subconfig = {k: v for k, v in self.config.__dict__.items() if k in ['model', 'source']}
        for conversation in tqdm(conversations, desc=str(subconfig)):
            history = conversation
            cols = os.get_terminal_size().columns
            print(str(history[0])[:cols])
            print(str(history[1])[:cols])

            initial_response = client.chat.completions.create(
                messages=history,
                model=self.model,
                max_tokens=512,
                temperature=self.config.temperature,
            )

            # Append model's response to conversation history
            response = initial_response.choices[0].message.content
            history.append({'role': 'assistant', 'content': response})
            print(str(history[-1])[:cols])

            # Append another user prompt (assuming it's provided in the DataFrame)
            history.append({'role': 'user', 'content': self.config.system_content[1]})
            print(str(history[-1])[:cols])

            # Generate final response
            final_response = client.chat.completions.create(
                messages=history,
                model=self.model,
                max_tokens=512,
                temperature=self.config.temperature,
            )

            response = final_response.choices[0].message.content
            history.append({'role': 'assistant', 'content': response})
            print(str(history[-1])[:cols])

            # concat last two assistant responses to get final response
            fixed_response = history[-3]['content'] + '\n\n' + history[-1]['content']

            full_conversations.append(history)
            parsed_final_responses.append(fixed_response)

        df['conversation'] = full_conversations
        df['parsed_response'] = parsed_final_responses

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
        # df['parsed_response'] = df['conversa'].apply(lambda row: +'\n\n' + row['final_response'], axis=1)
        # # clean up math
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

        for cfg in cfgs:
            print(f'Running with config:\n{json.dumps(cfg, indent=4)}')
            config = ConversationConfig(**cfg)
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
    parser.add_argument(
        '--random_seed', type=int, default=72, help='Random seed for generation. 72 is default for survey generation.'
    )
    parser.add_argument('--save', action='store_true', help='Save results.')
    parser.add_argument(
        '--source',
        type=str,
        default='Franklin/full_study.jsonl',
        help='Source data.',
        choices=SOURCES,
    )
    parser.add_argument(
        '--system-content',
        type=list,
        default=[
            'Create a step-by-step plan for finding the answer to the following problem. Do not answer the question. Do not perform the actions in the plan. Your only task is to outline the steps involved in a concise and clear manner.',
            'Now perform the steps in the plan you created. Use the most precise, accurate and up-to-date information available. To save space, be concise when describing the actions. Conclude by stating the answer that you reached by following the steps you outlined.',
        ],
        help='System content.',
    )
    parser.add_argument('--temperature', type=float, default=0.2, help='Generation temperature.')
    args = parser.parse_args()

    if args.batch:
        batch(args.batch)

    else:
        config = ConversationConfig(
            batch_size=args.batch_size,
            description=args.description,
            examples=args.num_of_examples,
            model=args.model,
            random_seed=args.random_seed,
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
