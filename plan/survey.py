"""
This module contains functions for formatting the output of the LLM responses.
"""

import argparse
import json
from pathlib import Path
import re
from uuid import uuid4

import pandas as pd


def concat_all_responses(
    files: list,
) -> pd.DataFrame:
    """
    Concatenate all LLM responses from the input files.

    Parameters
    ----------
    files (list)
        The input files from outputs/.

    Returns
    -------
    pd.DataFrame
        The concatenated LLM responses.
    """
    log = pd.read_json("outputs/log.jsonl", lines=True)

    all_responses_for_survey = pd.DataFrame()

    for file in files:
        df = pd.read_json(f"outputs/{file}", lines=True)
        # get dataset name
        dataset = (
            log[log["filename"] == file]["source"]
            .apply(lambda x: x.split("/")[0])
            .values[0]
        )
        new_df = pd.DataFrame()
        new_df["uuid"] = pd.Series(str(uuid4()) for i in range(len(df)))
        # new_df["internal_id"] = df["internal_id"]
        # fill column with len(df) * dataset
        new_df["dataset"] = pd.Series(dataset for i in range(len(df)))
        new_df["question"] = df["question"]
        new_df["parsed_responses"] = df["parsed_responses"]
        new_df["with_prefix"] = new_df.apply(
            lambda row: apply_template(row["question"], row["parsed_responses"]), axis=1
        )
        new_df["html"] = new_df.apply(
            lambda row: markdown_to_html(row["with_prefix"]), axis=1
        )

        all_responses_for_survey = pd.concat([all_responses_for_survey, new_df])

    return all_responses_for_survey


def markdown_to_html(
    markdown: str,
) -> str:
    """
    Convert markdown-formatted LLM responses to HTML.

    Parameters
    ----------
    markdown (str)
        The markdown-formatted LLM response.

    Returns
    -------
    str
        The HTML-formatted LLM response.
    """
    # replace **text** with <strong>text</strong>
    html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", markdown)
    # replace *text* with <em>text</em>
    html = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html)
    # replace # with <h1>
    html = re.sub(r"\n# (.*?)\n", r"<h1>\1</h1>", html)
    # replace ## with <h2>
    html = re.sub(r"\n## (.*?)\n", r"<h2>\1</h2>", html)
    # replace ### with <h3>
    html = re.sub(r"\n### (.*?)\n", r"<h3>\1</h3>", html)
    # replace #### with <h4>
    html = re.sub(r"\n#### (.*?)\n", r"<h4>\1</h4>", html)
    # replace ##### with <h5>
    html = re.sub(r"\n##### (.*?)\n", r"<h5>\1</h5>", html)
    # replace ###### with <h6>
    html = re.sub(r"\n###### (.*?)\n", r"<h6>\1</h6>", html)
    # replace \n with <br>
    html = re.sub(r"\n", r"<br>", html)
    # replace ```code``` with <pre><code>code</code></pre>
    html = re.sub(r"```(.*?)```", r"<pre><code>\1</code></pre>", html, flags=re.DOTALL)
    # replace `code` with <code>code</code>
    html = re.sub(r"`(.*?)`", r"<code>\1</code>", html)
    # replace [text](url) with <a href="url">text</a>
    html = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', html)
    # replace ![alt text](url) with <img src="url" alt="alt text">
    html = re.sub(r"!\[(.*?)\]\((.*?)\)", r'<img src="\2" alt="\1">', html)
    # replace - item or * item with <ul><li>item</li></ul>
    html = re.sub(r"(\n|^)[*-] (.*?)(\n|$)", r"\1<ul><li>\2</li></ul>\3", html)
    # replace 1. item with <ol><li>item</li></ol>
    html = re.sub(r"(\n|^)\d+\. (.*?)(\n|$)", r"\1<ol><li>\2</li></ol>\3", html)
    # replace > text with <blockquote>text</blockquote>
    html = re.sub(r"\n> (.*?)\n", r"<blockquote>\1</blockquote>", html)
    # replace --- or *** with <hr>
    html = re.sub(r"\n[-*]{3,}\n", r"<hr>", html)

    return html


def apply_template(
    question: str,
    response: str,
) -> str:
    """
    Apply the template to the question and response.

    Parameters
    ----------
    question (str)
        The question.
    response (str)
        The response.

    Returns
    -------
    str
        The formatted question and response.
    """
    return f"Question: {question}\n\n{response}"


def generate_outputs(
    input_file: str,
) -> pd.DataFrame:
    """
    Load the LLM responses from the input file.

    Parameters
    ----------
    input_file (str)
        The input file from outputs/.

    Returns
    -------
    pd.DataFrame
        The LLM responses.
    """
    df = pd.read_json(input_file, lines=True, orient="records")
    df = pd.DataFrame(df)
    df["with_prefix"] = df.apply(
        lambda row: apply_template(row["question"], row["parsed_responses"]), axis=1
    )
    df["html"] = df.apply(lambda row: markdown_to_html(row["with_prefix"]), axis=1)

    return df


class QualtricsSurvey:
    def __init__(
        self,
        inputs: pd.DataFrame,
    ):
        self.qid = 1
        self.inputs = inputs
        with open("survey_template.json", "r") as f:
            self.SURVEY = json.load(f)

        with open("survey_default_question.json", "r") as f:
            self.DEFAULT_QUESTION = json.load(f)

        with open("survey_default_block.json", "r") as f:
            self.BLOCKS = json.load(f)

    def add_question_to_survey(
        self,
        uuid: str,
        question: str,
    ):
        """
        Add a question to a Qualtrics survey.

        Parameters
        ----------
        question: dict
            The question.
        """
        new_question = self.DEFAULT_QUESTION
        new_question["PrimaryAttribute"] = f"QID{self.qid}"
        new_question["SecondaryAttribute"] = uuid
        new_question["Payload"]["QuestionText"] = question
        new_question["Payload"]["DataExportTag"] = f"QID{self.qid}"
        new_question["Payload"]["QuestionDescription"] = uuid
        new_question["Payload"]["QuestionID"] = f"QID{self.qid}"

        self.SURVEY["SurveyElements"].append(new_question)

    def add_question_to_block(
        self,
        dataset: str,
    ):
        """
        Add a question to a block of a Qualtrics survey.

        Parameters
        ----------
        dataset: str
            The block description/dataset name.
        """
        # Find the index of the block with the dataset name
        block_index = next(
            index
            for index, block in enumerate(self.BLOCKS["Payload"])
            if block["Description"] == dataset
        )

        # Create the new block element
        new_element = {
            "Type": "Question",
            "QuestionID": f"QID{self.qid}",
        }

        # Append the new element to the block's elements
        self.BLOCKS["Payload"][block_index]["BlockElements"].append(new_element)

    def fill(
        self,
        questions: pd.DataFrame,
    ):
        """
        Fill the survey with questions.

        Parameters
        ----------
        questions: pd.DataFrame
            The questions.
        """
        for i, row in questions.iterrows():
            self.add_question_to_survey(row["uuid"], row["html"])
            self.add_question_to_block(row["dataset"])
            self.qid += 1

        self.SURVEY["SurveyElements"].append(self.BLOCKS)

        with open("survey.json", "w") as f:
            json.dump(self.SURVEY, f, indent=4)

        with open("survey.qsf", "w") as f:
            f.write(json.dumps(self.SURVEY))


if __name__ == "__main__":
    files = [
        "2024-08-29T16:48:04.jsonl",
        "2024-08-29T16:50:12.jsonl",
        "2024-08-29T16:52:22.jsonl",
        "2024-08-29T16:53:34.jsonl",
        "2024-08-29T16:56:09.jsonl",
        "2024-08-29T16:56:41.jsonl",
        "2024-08-29T16:58:15.jsonl",
        "2024-08-29T16:58:59.jsonl",
        "2024-08-29T17:00:43.jsonl",
        "2024-08-29T17:03:19.jsonl",
        "2024-08-29T17:05:57.jsonl",
        "2024-08-29T17:08:30.jsonl",
    ]

    all_responses = concat_all_responses(files)
    QualtricsSurvey(all_responses).fill(all_responses)
