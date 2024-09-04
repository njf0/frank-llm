"""Module for formatting the output of the LLM responses."""

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pandas as pd

PWD = Path.cwd()
LOG_PATH = Path(PWD, 'plan', 'outputs', 'log').with_suffix('.jsonl')


def markdown_to_html(
    markdown: str,
) -> str:
    """Convert markdown-formatted LLM responses to HTML.

    Parameters
    ----------
    markdown: str
        The markdown-formatted LLM response.

    Returns
    -------
    str
        The HTML-formatted LLM response.

    """
    # replace **text** with <strong>text</strong>
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', markdown)
    # replace *text* with <em>text</em>
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    # replace # with <h1>
    html = re.sub(r'\n# (.*?)\n', r'<h1>\1</h1>', html)
    # replace ## with <h2>
    html = re.sub(r'\n## (.*?)\n', r'<h2>\1</h2>', html)
    # replace ### with <h3>
    html = re.sub(r'\n### (.*?)\n', r'<h3>\1</h3>', html)
    # replace #### with <h4>
    html = re.sub(r'\n#### (.*?)\n', r'<h4>\1</h4>', html)
    # replace ##### with <h5>
    html = re.sub(r'\n##### (.*?)\n', r'<h5>\1</h5>', html)
    # replace ###### with <h6>
    html = re.sub(r'\n###### (.*?)\n', r'<h6>\1</h6>', html)
    # replace \n with <br>
    html = re.sub(r'\n', r'<br>', html)
    # replace ```code``` with <pre><code>code</code></pre>
    html = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    # replace `code` with <code>code</code>
    html = re.sub(r'`(.*?)`', r'<code>\1</code>', html)
    # replace [text](url) with <a href="url">text</a>
    html = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', html)
    # replace ![alt text](url) with <img src="url" alt="alt text">
    html = re.sub(r'!\[(.*?)\]\((.*?)\)', r'<img src="\2" alt="\1">', html)
    # replace - item or * item with <ul><li>item</li></ul>
    html = re.sub(r'(\n|^)[*-] (.*?)(\n|$)', r'\1<ul><li>\2</li></ul>\3', html)
    # replace 1. item with <ol><li>item</li></ol>
    html = re.sub(r'(\n|^)\d+\. (.*?)(\n|$)', r'\1<ol><li>\2</li></ol>\3', html)
    # replace > text with <blockquote>text</blockquote>
    html = re.sub(r'\n> (.*?)\n', r'<blockquote>\1</blockquote>', html)
    # replace --- with <hr>
    html = re.sub(r'-{3}', r'<hr>', html)

    return html


def get_files_from_description(
    description: str,
) -> list:
    """Get the files from the description in the log file.

    Parameters
    ----------
    description: str
        The description of the files.

    Returns
    -------
    list
        The files from the description.

    """
    log = pd.read_json(LOG_PATH, lines=True)
    return log[log['description'] == description]['filename'].to_list()


def prepare_inputs(
    files: list | None = None,
) -> pd.DataFrame:
    """Concatenate all LLM responses from the input files.

    Parameters
    ----------
    files: list
        The input files from outputs/.

    Returns
    -------
    pd.DataFrame
        The concatenated LLM responses.

    """
    log = pd.read_json(Path(PWD, 'plan', 'outputs', 'log').with_suffix('.jsonl'), lines=True)

    all_inputs = pd.DataFrame()

    for file in files:
        df = pd.read_json(Path(PWD, 'plan', 'outputs', file), lines=True)
        # get dataset name
        dataset = log[log['filename'] == file]['source'].apply(lambda x: x.split('/')[0]).to_numpy()[0]
        new_df = pd.DataFrame()
        new_df['uuid'] = pd.Series(str(uuid4()) for i in range(len(df)))
        # new_df["internal_id"] = df["internal_id"]
        new_df['dataset'] = pd.Series(dataset for i in range(len(df)))
        new_df['question'] = df['question']
        new_df['parsed_responses'] = df['parsed_responses']
        new_df['with_prefix'] = new_df.apply(lambda row: f'*{row["question"]}*---{row["parsed_responses"]}', axis=1)
        new_df['html'] = new_df.apply(lambda row: markdown_to_html(row['with_prefix']), axis=1)

        all_inputs = pd.concat([all_inputs, new_df])

    return all_inputs


class QualtricsSurvey:
    """A class for creating a Qualtrics survey."""

    def __init__(
        self,
        inputs: pd.DataFrame,
        save_filename: str,
    ):
        """Initialize the Qualtrics survey.

        Parameters
        ----------
        inputs: pd.DataFrame
            The LLM responses.
        save_filename: str
            The filename to save the survey.

        """
        self.qid = 1
        self.inputs = inputs
        self.save_filename = save_filename

        with Path(PWD, 'plan', 'survey', 'survey_template').with_suffix('.json').open() as f:
            self.SURVEY = json.load(f)

        with Path(PWD, 'plan', 'survey', 'question_template').with_suffix('.json').open() as f:
            self.DEFAULT_QUESTION = json.load(f)

        with Path(PWD, 'plan', 'survey', 'block_template').with_suffix('.json').open() as f:
            self.BLOCKS = json.load(f)

    def add_question_to_survey(
        self,
        uuid: str,
        question: str,
    ):
        """Add a question to a Qualtrics survey.

        Parameters
        ----------
        uuid: str
            The UUID of the question/model/dataset combination.
        question: dict
            The question from a given dataset.

        """
        new_question = deepcopy(self.DEFAULT_QUESTION)
        new_question['PrimaryAttribute'] = f'QID{self.qid}'
        new_question['SecondaryAttribute'] = uuid
        new_question['Payload']['QuestionText'] = question
        new_question['Payload']['DataExportTag'] = uuid
        new_question['Payload']['QuestionDescription'] = uuid
        new_question['Payload']['QuestionID'] = f'QID{self.qid}'

        self.SURVEY['SurveyElements'].append(new_question)

    def add_question_to_block(
        self,
        dataset: str,
    ):
        """Add a question to a block of a Qualtrics survey.

        Parameters
        ----------
        dataset: str
            The block description/dataset name.

        """
        # Find the index of the block with the dataset name
        block_index = next(index for index, block in enumerate(self.BLOCKS['Payload']) if block['Description'] == dataset)

        # Create the new block element
        new_element = {
            'Type': 'Question',
            'QuestionID': f'QID{self.qid}',
        }

        # Append the new element to the block's elements
        self.BLOCKS['Payload'][block_index]['BlockElements'].append(new_element)

    def fill(
        self,
        questions: pd.DataFrame,
    ):
        """Fill the survey with questions.

        Parameters
        ----------
        questions: pd.DataFrame
            The questions.

        """
        for _, row in questions.iterrows():
            self.add_question_to_survey(row['uuid'], row['html'])
            self.add_question_to_block(row['dataset'])
            self.qid += 1

        self.SURVEY['SurveyElements'].append(self.BLOCKS)

        with Path(PWD, 'plan', 'survey', f'{self.save_filename}').with_suffix('.json').open('w') as f:
            json.dump(self.SURVEY, f, indent=4)

        with Path(PWD, 'plan', 'survey', f'{self.save_filename}').with_suffix('.qsf').open('w') as f:
            f.write(json.dumps(self.SURVEY))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare inputs for a Qualtrics survey.')

    parser.add_argument('--description', type=str, help='The description of the files.')
    parser.add_argument('--save_filename', default='survey', type=str, help='The filename to save the survey.')
    args = parser.parse_args()

    files = get_files_from_description(args.description)
    inputs = prepare_inputs(files)
    print(inputs.sample(10))
    survey = QualtricsSurvey(inputs, args.save_filename)
    survey.fill(inputs)
