"""Module for formatting the output of the LLM responses."""

import argparse
import json
import random
import re
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pandas as pd
from markdown import markdown

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
    return log[log['description'] == description]


def insert_attention_checks(
    description: str,
) -> pd.DataFrame:
    """Add attention checks to the survey."""
    log = pd.read_json(Path(PWD, 'plan', 'outputs', 'log').with_suffix('.jsonl'), lines=True)
    ac_file = log[log['description'] == description]['filename'].to_numpy()[0]

    df = pd.read_json(Path(PWD, 'plan', 'outputs', ac_file), lines=True)
    if 'agree' in description:
        ac_text = 'To show that I am paying attention, I will select "strongly agree" as my answers for this question.'
    elif 'disagree' in description:
        ac_text = 'To show that I am paying attention, I will select "strongly disagree" as my answers for this question.'
    else:
        raise ValueError('Description must contain "agree" or "disagree".')

    def insert_ac(df, parsed_responses, ac_text):
        """Insert attention check into a random position in sentences of row['parsed_responses']."""
        sentences = parsed_responses.split('\n')
        random_index = random.randint(0, len(sentences) - 1)
        sentences.insert(random_index, ac_text)
        return '\n'.join(sentences)

    df['parsed_responses'] = df['parsed_responses'].apply(lambda row: insert_ac(df, row, ac_text))

    return df


def prepare_inputs(
    files: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate all LLM responses from the input files.

    Parameters
    ----------
    files: pd.DataFrame
        The input files from outputs/.

    Returns
    -------
    pd.DataFrame
        The concatenated LLM responses.

    """
    log = pd.read_json(Path(PWD, 'plan', 'outputs', 'log').with_suffix('.jsonl'), lines=True)

    def process_file(file):
        df = pd.read_json(Path(PWD, 'plan', 'outputs', file), lines=True)
        dataset = log[log['filename'] == file]['source'].apply(lambda x: x.split('/')[0]).to_numpy()[0]
        model = log[log['filename'] == file]['model'].to_numpy()[0]
        new_df = pd.DataFrame(
            {
                'uuid': [str(uuid4()) for _ in range(len(df))],
                'is_attention_check': False,
                'dataset': dataset,
                'model': model,
                'question': df['question'],
                'parsed_responses': df['parsed_responses'],
            }
        )
        new_df['html'] = new_df['parsed_responses'].apply(markdown)
        new_df['html'] = new_df.apply(lambda row: f'<p><em>{row["question"]}</em></p><hr>{row["html"]}', axis=1)
        return new_df

    all_inputs = pd.concat(files['filename'].apply(process_file).tolist(), ignore_index=True)

    return all_inputs


class QualtricsSurvey:
    """A class for creating a Qualtrics survey."""

    def __init__(
        self,
    ):
        """Initialize the Qualtrics survey."""
        self.qid = 1
        self.save_path = Path(PWD, 'plan', 'survey')

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
        block_name: str,
    ):
        """Add a question to a block of a Qualtrics survey.

        Parameters
        ----------
        block_name: str
            The block description/dataset name.

        """
        # Find the index of the block with the dataset name
        block_index = next(index for index, block in enumerate(self.BLOCKS['Payload']) if block['Description'] == block_name)

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
        attention_checks: tuple | None = None,
        save_filename: str = 'survey',
    ) -> pd.DataFrame:
        """Fill the survey with questions.

        Parameters
        ----------
        questions: pd.DataFrame
            The questions.
        attention_checks: tuple, optional
            The attention checks.

        """
        for _, row in questions.iterrows():
            print(f'Adding question {self.qid} to survey...', end='\r')
            self.add_question_to_survey(row['uuid'], row['html'])
            self.add_question_to_block(row['dataset'])
            self.qid += 1

        agree, disagree = attention_checks

        for _, row in agree.iterrows():
            print(f'Adding question {self.qid} to survey...', end='\r')
            self.add_question_to_survey(row['uuid'], row['html'])
            self.add_question_to_block('attention-check-agree')
            self.qid += 1

        for _, row in disagree.iterrows():
            print(f'Adding question {self.qid} to survey...', end='\r')
            self.add_question_to_survey(row['uuid'], row['html'])
            self.add_question_to_block('attention-check-disagree')
            self.qid += 1

        self.SURVEY['SurveyElements'].append(self.BLOCKS)

        print('\nSaving survey...')
        with Path(self.save_path, f'{self.save_filename}').with_suffix('.json').open('w') as f:
            json.dump(self.SURVEY, f, indent=4)

        with Path(self.save_path, f'{self.save_filename}').with_suffix('.qsf').open('w') as f:
            f.write(json.dumps(self.SURVEY))

        # also save the dataframe as jsonl
        questions.to_json(Path(self.save_path, f'{self.save_filename}').with_suffix('.jsonl'), orient='records', lines=True)

        return questions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare inputs for a Qualtrics survey.')

    parser.add_argument('--description', type=str, help='The description of the files.')
    parser.add_argument('--save', default='survey', type=str, help='The filename to save the survey.')
    args = parser.parse_args()

    print('Getting files...')
    files = get_files_from_description(args.description)
    print(files)
    print('Preparing attention checks...')
    agree = insert_attention_checks('attention-check-agree')
    disagree = insert_attention_checks('attention-check-disagree')
    print('Preparing inputs...')
    inputs = prepare_inputs(files)
    survey = QualtricsSurvey()
    questions = survey.fill(inputs, attention_checks=(agree, disagree), save_filename=args.save)
    print(questions.sample(10))
    print(f'Survey saved to {survey.save_path}/{survey.save_filename}.json!')
