"""Module for loading and processing datasets."""

from pathlib import Path

import pandas as pd

PWD = Path.cwd()
DATA_DIR = Path(PWD, 'resources', 'data')
CONFIG = {
    'StrategyQA': {
        'url': 'https://huggingface.co/datasets/njf/StrategyQA/resolve/main/',
        'defaults': ['train.jsonl', 'dev.jsonl'],
        'columns': ['qid', 'question'],
    },
    'HotpotQA': {
        'url': 'https://huggingface.co/datasets/njf/HotpotQA/resolve/main/',
        'defaults': ['hotpot_test_fullwiki_v1.json'],
        'columns': ['_id', 'question'],
    },
    'GSM8k': {
        'url': 'https://huggingface.co/datasets/njf/GSM8k/resolve/main/',
        'defaults': ['train.jsonl', 'test.jsonl'],
        'columns': ['question'],
    },
    'Franklin': {
        'url': 'https://huggingface.co/datasets/njf/Franklin/resolve/main/',
        'defaults': ['full_study.jsonl'],
        'columns': ['internal_id', 'question'],
    },
}


class Dataset:
    """Class for loading and processing datasets."""

    def __init__(
        self,
        dataset_name: str,
        source: str,
    ) -> None:
        """Initialize the dataset, source, full path, url, and columns.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset.
        source : str
            The source of the dataset.

        """
        self.dataset_name = dataset_name
        self.source = Path(source)
        self.full_path = DATA_DIR / self.source
        self.url = CONFIG[dataset_name]['url'] + self.source.name
        self.columns = CONFIG[dataset_name]['columns']

    def load_data(self) -> pd.DataFrame:
        """Load the dataset."""
        if not self.full_path.exists():
            df = pd.read_json(self.url, lines=True)
            df = pd.DataFrame(df)
            self.full_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_json(self.full_path, lines=True, orient='records')
        else:
            df = pd.read_json(self.full_path, lines=True)

        return df

    def process_data(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Process the dataset."""
        return df[self.columns]

    def __call__(self) -> pd.DataFrame:
        """Load and process the dataset."""
        df = self.load_data()
        return self.process_data(df)


if __name__ == '__main__':
    dataset = Dataset('StrategyQA', 'dev.jsonl')
    print(dataset())
