"""Module for loading and processing datasets."""

from pathlib import Path

import pandas as pd

PWD = Path.cwd()
DATA_DIR = Path(PWD, 'resources', 'data')
CONFIG = {
    'StrategyQA': {
        'url': 'https://huggingface.co/datasets/njf/StrategyQA/resolve/main/',
        'defaults': ['train.jsonl', 'dev.jsonl', 'dev_decomp_max.jsonl', 'dev_decomp_3_plus.jsonl', 'dev_decomp_4_plus.jsonl'],
        'columns': {'internal_id': 'qid', 'question': 'question'},
    },
    'HotpotQA': {
        'url': 'https://huggingface.co/datasets/njf/HotpotQA/resolve/main/',
        'defaults': ['hotpot_test_fullwiki_v1.jsonl'],
        'columns': {'internal_id': '_id', 'question': 'question'},
    },
    'GSM8k': {
        'url': 'https://huggingface.co/datasets/njf/GSM8k/resolve/main/',
        'defaults': ['train.jsonl', 'test.jsonl'],
        'columns': {'question': 'question'},
    },
    'Franklin': {
        'url': 'https://huggingface.co/datasets/njf/Franklin/resolve/main/',
        'defaults': ['full_study.jsonl'],
        'columns': {'internal_id': 'internal_id', 'question': 'question'},
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
        self.full_path = DATA_DIR / self.dataset_name / self.source
        self.url = CONFIG[dataset_name]['url'] + self.source.name
        self.columns = CONFIG[dataset_name]['columns']

    def load_data(self) -> pd.DataFrame:
        """Load the dataset."""
        if not self.full_path.exists():
            # print(f'Dataset {self.full_path} not found. Downloading and saving...', end=' ')
            df = pd.read_json(self.url, lines=True)
            df = pd.DataFrame(df)
            self.full_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_json(Path(DATA_DIR, self.dataset_name, self.source), orient='records', lines=True)
            # print('done.')
        else:
            # print(f'Loading dataset from {self.full_path}...', end=' ')
            df = pd.read_json(self.full_path, lines=True)
            # print('done.')

        return df[self.columns.values()].rename(columns=self.columns)


if __name__ == '__main__':
    # download all datasets
    for dataset_name, config in CONFIG.items():
        for source in config['defaults']:
            # check if file already exists
            if (DATA_DIR / dataset_name / source).exists():
                print(f'{dataset_name}/{source} already downloaded.')
            else:
                print(f'Downloading {dataset_name}/{source}...', end=' ')
                dataset = Dataset(dataset_name, source).load_data()
                print('done.')
