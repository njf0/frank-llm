'''
This module contains the base classes for generating responses using LLMs (Language Model Models) on a dataset of Frank-style questions.

The module includes the following classes:

- `GenerationConfig`: A class representing the configuration for generating responses.
- `GenerationBase`: A base class for applying LLMs to a dataset of Frank-style questions.

The module also defines a dictionary `CONFIG_TYPES` that maps configuration keys to their corresponding types.

Note: This module assumes the presence of the `pandas` library for handling dataframes and the `json` library for working with JSON files.
'''
import datetime
import json
import random
from pathlib import Path

import pandas as pd

CONFIG_TYPES = {
    'batch_size': int,
    'examples': int,
    'description': str,
    'model': str,
    'temperature': float,
    'source': str,
    'system_content': str
}

class GenerationConfig:
    '''
    config = {
        "batch_size": 16,
        "description": "Test addition of description column.",
        "examples": 16,
        # "model": "/nfs/public/hf/models/meta-llama/Meta-Llama-3-8B-Instruct",
        'model': 'mistralai/Mistral-7B-Instruct-v0.3',
        "temperature": 0.2,
        "source": "/app/resources/data/full_study.json",
        "system_content": "Answer the following question.",
    }
    '''
    def __init__(
            self,
    ) -> None:
        self.batch_size: int = 16
        self.examples: int = -1
        self.description: str = ''
        self.model: str = ''
        self.temperature: float = 0.2,
        self.source: str = '/app/resources/data/full_study.json'
        self.system_content: str = ''



class GenerationBase:
    """
    Base class for applying LLMs to dataset of Frank-style questions.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model and data parameters.
    """
    def __init__(
            self,
            cfg: dict,
    ) -> None:

        # type-check config against CONFIG_TYPES
        for key, value in cfg.items():
            if key not in CONFIG_TYPES:
                raise ValueError(f"Invalid key in config: {key}")
            if not isinstance(value, CONFIG_TYPES[key]):
                raise ValueError(f"Invalid type for '{key}' ({type(value)}). Should be {CONFIG_TYPES[key]}.")

        # make sure all keys are present
        if cfg.keys() != CONFIG_TYPES.keys():
            # report missing keys
            missing_keys = set(CONFIG_TYPES.keys()) - set(cfg.keys())
            raise ValueError(f"Missing keys in config: {', '.join(missing_keys)}")

        self.config = cfg

    def load_inputs(
            self,
    ) -> list:
        """
        Load inputs from source file as given in config.
        """
        inputs = Path(self.config["source"])
        with inputs.open("r", encoding="utf-8") as f:
            inputs = [v["question"] for v in random.sample(list(json.load(f).values()), self.config["examples"])]

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
