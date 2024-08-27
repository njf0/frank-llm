'''
Class for loading datasets from huggingface repos
'''
import json
from pathlib import Path

import pandas as pd


class StrategyQA():
    """
    Class for loading and processing the Franklin dataset.
    """
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

        if self.dataset != 'strategyqa':
            raise ValueError(f"Dataset '{self.dataset}' not recognized.")

    def load_data(
            self,
    ) -> pd.DataFrame:
        """
        Load the StrategyQA dataset.
        """
        # check if the dataset is already downloaded
        if not self.path.exists():

            df = pd.read_json(f'https://huggingface.co/datasets/njf/StrategyQA/resolve/main/{self.file}', lines=True)
            df = pd.DataFrame(df)
            Path('/app/resources/data/strategyqa').mkdir(parents=True)
            df.to_json(f'/app/resources/data/strategyqa/{self.file}', lines=True, orient='records')

        else:
            df = pd.read_json(f'/app/resources/data/strategyqa/{self.file}', lines=True)


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
        Initialize the HotpotQA dataset.
        """
        self.file = file
        self.name = 'hotpotqa'

    def load_data(
            self,
    ) -> pd.DataFrame:
        """
        Load the HotpotQA dataset.
        """
        # check if the dataset is already downloaded
        if not Path('/app/resources/data/hotpotqa').exists():

            df = pd.read_json(f'https://huggingface.co/datasets/HotpotQA/resolve/main/{self.file}.json')
            df = pd.DataFrame(df)

        else:
            df = pd.read_json(f'/app/resources/data/hotpotqa/{self.file}.jsonl', lines=True, orient='records')

        return df

    def save_data(
            self,
            df: pd.DataFrame,
    ) -> None:
        """
        Save the HotpotQA dataset.
        """
        Path('/app/resources/data/hotpotqa').mkdir()
        df.to_json(f'/app/resources/data/hotpotqa/{self.file}.jsonl', lines=True)

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