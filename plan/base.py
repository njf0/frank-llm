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
from data import Dataset

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
        self.filename = ''

    def load_inputs(self) -> list:
        """
        Load inputs from source file as given in config.
        """
        dataset_name = Path(self.config["source"]).parent.name
        source = self.config["source"]

        if dataset_name not in Dataset.CONFIG:
            raise ValueError(f"Dataset '{dataset_name}' not recognized.")

        dataset = Dataset(dataset_name, source)

        return dataset()

    def save_results(
            self,
            df: pd.DataFrame,
    ) -> None:
        """
        Save results to output file.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing results.
        """
        if not Path('/app/plan/outputs').exists():
            Path('/app/plan/outputs').mkdir()
        logfile = Path('/app/plan/outputs/log.csv')
        if not logfile.exists():
            with open(logfile, 'w', encoding='utf-8') as f:
                log = pd.DataFrame()
                log.to_csv(f, index=True)

        self.filename = datetime.datetime.isoformat(datetime.datetime.now())

        df.to_csv(f"/app/plan/outputs/{self.filename}.csv", index=False)

        with open('/app/plan/outputs/log.csv', 'r', encoding='utf-8') as f:
            log = pd.read_csv(f, index_col=0)

        new_log_entry_df = pd.DataFrame(columns=self.config.keys(), data=[self.config.values()], index=[self.filename])
        log = pd.concat([log, new_log_entry_df])

        with open('/app/plan/outputs/log.csv', 'w', encoding='utf-8') as f:
            log.to_csv(f, index=True)

        return df
