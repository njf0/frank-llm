"""
This module contains functions for formatting the output of the LLM responses.
"""

import argparse
import json
import re


import pandas as pd


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert markdown-formatted LLM responses to HTML."
    )
    parser.add_argument("--input", type=str, help="Input file from outputs/")

    args = parser.parse_args()
