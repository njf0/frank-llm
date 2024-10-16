"""Module for formatting the output of the LLM responses."""

import argparse
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pandas as pd
from markdown import markdown

PWD = Path.cwd()

# def insert_attention_checks(
#     description: str,
# ) -> pd.DataFrame:
#     """Add attention checks to the survey."""
#     log = get_log()
#     ac_file = log[log['description'] == description]

#     df = pd.read_json(Path(PWD, 'mini-project', 'outputs', ac_file['filename'].to_numpy()[0]), lines=True)
#     if 'agree' in description:
#         ac_text = 'To show that I am paying attention, I will select "strongly agree" as my answers for this question.'
#     elif 'disagree' in description:
#         ac_text = 'To show that I am paying attention, I will select "strongly disagree" as my answers for this question.'
#     else:
#         raise ValueError('Description must contain "agree" or "disagree".')

#     def insert_ac(df, parsed_responses, ac_text):
#         """Insert attention check into a random position in sentences of row['parsed_response']."""
#         sentences = parsed_responses.split('\n')
#         random_index = random.randint(0, len(sentences) - 1)
#         sentences.insert(random_index, ac_text)
#         return '\n'.join(sentences)

#     df['parsed_response'] = df['parsed_response'].apply(lambda row: insert_ac(df, row, ac_text))

#     new_df = pd.DataFrame(
#         {
#             'uuid': [str(uuid4()) for _ in range(len(df))],
#             'is_attention_check': True,
#             'dataset': ac_file['source'].to_numpy()[0],
#             'model': ac_file['model'].to_numpy()[0],
#             'question': df['question'],
#             'parsed_response': df['parsed_response'],
#             'html': df['parsed_response'].apply(markdown),
#         }
#     )

#     new_df['html'] = new_df.apply(lambda row: f'<p><em>{row["question"]}</em></p><hr>{row["html"]}', axis=1)

#     return new_df


class QualtricsSurvey:
    """A class for creating a (basic) Qualtrics survey."""

    def __init__(
        self,
        source: str,
    ):
        """Initialize the Qualtrics survey."""
        self.qid = 1
        self.source = source

        self.save_directory = Path(PWD, 'mini-project', 'studies', self.source, 'surveys')
        self.log_path = Path(PWD, 'mini-project', 'studies', self.source, 'output', 'log').with_suffix('.jsonl')

        with Path(PWD, 'mini-project', 'studies', self.source, 'surveys', 'survey_template').with_suffix('.json').open() as f:
            self.survey = json.load(f)

        with Path(PWD, 'mini-project', 'studies', self.source, 'surveys', 'question_template').with_suffix('.json').open() as f:
            self.question_template = json.load(f)

        with Path(PWD, 'mini-project', 'studies', self.source, 'surveys', 'block_template').with_suffix('.json').open() as f:
            self.blocks = json.load(f)

    def get_log(
        self,
    ) -> pd.DataFrame:
        """Get the log file."""
        return pd.read_json(self.log_path, lines=True)

    def get_files_from_description(
        self,
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
        log = pd.read_json(self.log_path, lines=True)
        return log[log['description'] == description]

    def format_qualtrics_question(
        self,
        question: str,
        answer: str,
    ) -> str:
        """Format a question for Qualtrics.

        Parameters
        ----------
        question: str
            The question to format.
        answer: str
            The answer to the question. Must be formatted as HTML, not Markdown.

        Returns
        -------
        str
            The formatted question.

        """
        return f'<p><em>{question}</em></p><hr>{answer}'

    def process_files(
        self,
        files: pd.DataFrame,
        length: int = -1,
    ) -> pd.DataFrame:
        """Concatenate all LLM responses from the input files.

        Parameters
        ----------
        files: pd.DataFrame
            The input files from outputs/.
        length: int, optional
            The number of questions to add to the survey.

        Returns
        -------
        pd.DataFrame
            The concatenated LLM responses.

        """
        log = self.get_log()

        def process_file(file):
            df = pd.read_json(Path(PWD, 'mini-project', 'studies', 'plan', 'output', file), lines=True)
            df = df.sample(length) if length > 0 else df

            dataset = log[log['filename'] == file]['source'].apply(lambda x: x.split('/')[0]).to_numpy()[0]
            model = log[log['filename'] == file]['model'].to_numpy()[0]

            survey_questions = pd.DataFrame(
                {
                    'uuid': [str(uuid4()) for _ in range(len(df))],
                    'is_attention_check': False,
                    'dataset': dataset,
                    'model': model,
                    'question': df['question'],
                    'parsed_response': df['parsed_response'],
                }
            )

            survey_questions['html'] = survey_questions['parsed_response'].apply(markdown)
            survey_questions['html'] = survey_questions.apply(
                lambda row: f'<p><em>{row["question"]}</em></p><hr>{row["html"]}', axis=1
            )

            return survey_questions

        all_inputs = pd.concat(files['filename'].apply(process_file).tolist(), ignore_index=True)

        return all_inputs

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
        new_question = deepcopy(self.question_template)
        new_question['PrimaryAttribute'] = f'QID{self.qid}'
        new_question['SecondaryAttribute'] = uuid
        new_question['Payload']['QuestionText'] = question
        new_question['Payload']['DataExportTag'] = uuid
        new_question['Payload']['QuestionDescription'] = uuid
        new_question['Payload']['QuestionID'] = f'QID{self.qid}'

        self.survey['SurveyElements'].append(new_question)

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
        try:
            block_index = next(
                index for index, block in enumerate(self.blocks['Payload']) if block['Description'] == block_name
            )
        except StopIteration:
            raise ValueError(f'Block "{block_name}" not found.') from StopIteration

        # Create the new block element
        new_element = {
            'Type': 'Question',
            'QuestionID': f'QID{self.qid}',
        }

        # Append the new element to the block's elements
        self.blocks['Payload'][block_index]['BlockElements'].append(new_element)

        # Add QID to random subset
        self.blocks['Payload'][block_index]['Options']['Randomization']['Advanced']['RandomSubSet'].append(f'QID{self.qid}')

    def fill(
        self,
        questions: pd.DataFrame,
        attention_checks: tuple | None = None,
    ) -> pd.DataFrame:
        """Fill the survey with questions.

        Parameters
        ----------
        questions: pd.DataFrame
            The questions.
        attention_checks: tuple, optional
            The attention checks.
        save: str, optional
            The filename to save the survey.

        """
        for _, row in questions.iterrows():
            print(f'Adding question {self.qid} to survey...', end='\r')
            self.add_question_to_survey(row['uuid'], row['html'])
            self.add_question_to_block(f'{row["dataset"]}*{row["model"].replace("/", "_")}')
            self.qid += 1

        # agree, disagree = attention_checks

        # for _, row in agree.iterrows():
        #     print(f'Adding question {self.qid} to survey...', end='\r')
        #     self.add_question_to_survey(row['uuid'], row['html'])
        #     self.add_question_to_block('attention-check-agree')
        #     self.qid += 1

        # for _, row in disagree.iterrows():
        #     print(f'Adding question {self.qid} to survey...', end='\r')
        #     self.add_question_to_survey(row['uuid'], row['html'])
        #     self.add_question_to_block('attention-check-disagree')
        #     self.qid += 1

        self.survey['SurveyElements'].append(self.blocks)

    def save_survey(
        self,
        questions: pd.DataFrame,
        save: str = 'survey',
    ) -> pd.DataFrame:
        """Save the survey to a JSON file.

        Parameters
        ----------
        save: str, optional
            The filename to save the survey.

        """
        print('\nSaving survey...')
        with Path(self.save_directory, f'{save}').with_suffix('.json').open('w') as f:
            json.dump(self.survey, f, indent=4)

        with Path(self.save_directory, f'{save}').with_suffix('.qsf').open('w') as f:
            f.write(json.dumps(self.survey))

        # also save the dataframe as jsonl
        # df = pd.concat([qestions, agree, disagree], ignore_index=True)

        questions.to_json(Path(self.save_directory, f'{save}').with_suffix('.jsonl'), orient='records', lines=True)

        return questions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare inputs for a Qualtrics survey.')

    parser.add_argument('--description', type=str, help='The description of the files.')
    parser.add_argument('--source', type=str, choices=['ask', 'plan'], help='Directory containing templates.')
    parser.add_argument('--save', default='survey', type=str, help='The filename to save the survey.')
    parser.add_argument('--length', default=-1, type=int, help='The number of questions to add to the survey.')
    args = parser.parse_args()

    Q = QualtricsSurvey(args.source)
    print('Getting files...')
    input_files = Q.get_files_from_description(args.description)
    print(input_files)
    # print('Preparing attention checks...')
    # agree = insert_attention_checks('attention-check-agree')
    # disagree = insert_attention_checks('attention-check-disagree')
    print('Preparing inputs...')
    all_input_questions = Q.process_files(input_files, length=args.length)
    Q.fill(all_input_questions)
    survey = Q.save_survey(all_input_questions, args.save)

    print(survey.sample(10))
    print(f'Survey saved to {Q.save_directory}/{args.save}.json!')
