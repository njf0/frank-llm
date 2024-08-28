import json
from pathlib import Path

import pandas as pd


class Dataset:
    '''
    Class for loading and processing datasets.
    '''
    DATA_DIR = Path('/app/resources/data')
    CONFIG = {
        'StrategyQA': {
            'url': 'https://huggingface.co/datasets/njf/StrategyQA/resolve/main/',
            'columns': ['qid', 'question']
        },
        'HotpotQA': {
            'url': 'https://huggingface.co/datasets/njf/HotpotQA/resolve/main/',
            'columns': ['_id', 'question']
        },
        'GSM8k': {
            'url': 'https://huggingface.co/datasets/njf/GSM8k/resolve/main/',
            'columns': ['question']
        },
        'Franklin': {
            'url': 'https://huggingface.co/datasets/njf/Franklin/resolve/main/',
            'columns': ['qid', 'question']
        }
    }

    def __init__(self, dataset_name: str, source: str) -> None:
        '''
        Initialize the dataset.
        '''
        self.dataset_name = dataset_name
        self.source = Path(source)
        self.full_path = self.DATA_DIR / self.source
        self.url = self.CONFIG[dataset_name]['url'] + self.source.name
        self.columns = self.CONFIG[dataset_name]['columns']

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset.
        """
        if not self.full_path.exists():
            df = pd.read_json(self.url, lines=True)
            df = pd.DataFrame(df)
            self.full_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_json(self.full_path, lines=True, orient='records')
        else:
            df = pd.read_json(self.full_path, lines=True)

        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the dataset.
        """
        return df[self.columns]

    def __call__(self) -> pd.DataFrame:
        """
        Load and process the dataset.
        """
        df = self.load_data()
        return self.process_data(df)


if __name__ == '__main__':
    dataset = Dataset('StrategyQA', 'dev.jsonl')
    print(dataset())