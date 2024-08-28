'''
Class for loading datasets from huggingface repos
'''
import json
from pathlib import Path

import pandas as pd


class Dataset:
    '''
    Base class for loading datasets from huggingface repos/locally
    '''
    def __init__(
            self,
            source: str,
    ) -> None:
        '''
        Initialize the dataset.
        '''
        self.DATA_DIR = Path('/app/resources/data')
        self.source = Path(source)
        self.dataset = self.source.parent.parts[-1]
        self.full_path = self.DATA_DIR / self.source


class StrategyQA(Dataset):
    """
    Class for loading and processing the Franklin dataset.
    """
    def __init__(
            self,
            source: str,
    ) -> None:
        """
        Initialize the StrategyQA dataset.
        """
        super().__init__(source)

        if self.dataset != 'StrategyQA':
            raise ValueError(f"Dataset '{self.dataset}' not recognized.")

    def load_data(
            self,
    ) -> pd.DataFrame:
        """
        Load the StrategyQA dataset.
        """
        # check if the dataset is already downloaded
        if not self.full_path.exists():

            df = pd.read_json(f'https://huggingface.co/datasets/njf/StrategyQA/resolve/main/{self.source.name}', lines=True)
            df = pd.DataFrame(df)
            self.full_path.parent.mkdir(parents=True)
            df.to_json(self.full_path, lines=True, orient='records')

        else:
            df = pd.read_json(self.full_path, lines=True)


        return df

    def process_data(
            self,
            df: pd.DataFrame,
    ) -> list:
        """
        Process the StrategyQA dataset.
        """
        # return qid and question cols in one df
        return df[['qid', 'question']]

    def __call__(
            self,
    ) -> list:
        """
        Load and process the StrategyQA dataset.
        """
        df = self.load_data()
        return self.process_data(df)

class HotpotQA:

    def __init__(
            self,
            file: str,
    ) -> None:
        """
        Initialize the StrategyQA dataset.
        """
        self.path = Path(file)
        self.dataset = self.path.parent.parts[-1]
        self.file = self.path.name

        if self.dataset != 'HotpotQA':
            raise ValueError(f"Dataset '{self.dataset}' not recognized.")

    def load_data(
            self,
    ) -> pd.DataFrame:
        """
        Load the HotpotQA dataset.
        """
        # check if the dataset is already downloaded
        if not self.path.exists():

            df = pd.read_json(f'https://huggingface.co/datasets/njf/HotpotQA/resolve/main/{self.file}', lines=True)
            df = pd.DataFrame(df)
            Path('/app/resources/data/HotpotQA').mkdir(parents=True)
            df.to_json(f'/app/resources/data/HotpotQA/{self.file}', lines=True, orient='records')

        else:
            df = pd.read_json(f'/app/resources/data/HotpotQA/{self.file}', lines=True, orient='records')

        return df

    def process_data(
            self,
            df: pd.DataFrame,
    ) -> list:
        """
        Process the HotpotQA dataset.
        """
        # return qid and question cols in one df
        return df[['_id', 'question']]

    def __call__(
            self,
    ) -> list:
        """
        Load and process the HotpotQA dataset.
        """
        df = self.load_data()
        return self.process_data(df)


class Franklin:
    """
    Class for loading and processing the Franklin dataset.
    """
    def __init__(
            self,
            file: str,
    ) -> None:
        """
        Initialize the Franklin dataset.
        """
        self.file = file
        self.name = 'franklin'

    def check_exists(
            self,
    ) -> bool:
        """
        Check if the Franklin dataset is already downloaded.
        """
        return Path('/app/resources/data/franklin').exists()

    def load_data(
            self,
    ) -> pd.DataFrame:
        """
        Load the Franklin dataset.
        """
        with self.file.open("r", encoding="utf-8") as f:
            inputs = [(k, v["question"]) for k, v in list(json.load(f).items())]

        # make df from inputs
        df = pd.DataFrame(inputs, columns=['qid', 'question'])

        return df

    def __call__(
            self,
    ) -> list:
        """
        Load and process the Franklin dataset.
        """
        return self.load_data()


if __name__ == '__main__':

    s = StrategyQA('dev.jsonl')
    print(s())