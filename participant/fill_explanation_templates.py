"""Fill explanation templates with subjects, properties, objects etc."""
import argparse
import json
import random
from pathlib import Path

import pandas as pd

import generate_data

with Path("resources/slot_values.json").open(encoding="utf-8") as f:
    SLOT_VALUES = json.load(f)

with Path("resources/iso-3166.csv").open(encoding="utf-8") as f:
    ISO_3166 = pd.read_csv(f)

DATA_GENERATORS = {
    'A1': generate_data.A1,
    'B1': generate_data.B1,
    'C1': generate_data.C1,
    'D1': generate_data.D1,
}

def load_template(
        template_name: str,
    ) -> tuple:
    """
    Load an explantion template.

    Parameters
    ----------
    template_name : str
        The name of the explanation template.

    Returns
    -------
    list
        The explanation template.
    """
    template_path = Path('resources', 'templates', f'{template_name}').with_suffix('.json')
    with open(template_path, encoding='utf-8') as f:
        template = json.load(f)

    question = random.choice(template['question_template'])
    required_entities = template['required_entities']
    explanation = template['explanation']

    return question, required_entities, explanation

def populate_entities(
        required_entities: list,
    ) -> dict:
    """
    Generate slot values for an explanation template.

    Parameters
    ----------
    values : str
        The values to fill the slots with.

    Returns
    -------
    dict
        The slot values in the form {'name': [v1, v2, ..., vn]}.
    """

    data = {}
    for entity in required_entities:
        if entity in SLOT_VALUES:
            data[entity] = random.sample(SLOT_VALUES[entity], 1)[0]
        elif entity == 'country':
            data[entity] = random.choice(list(set(ISO_3166['name'])))
        elif entity == 'region':
            data[entity] = random.choice(list(set(ISO_3166['sub-region'])))
        elif entity in ['country1', 'country2']:
            data[entity] = random.choice(ISO_3166['name'])

    if 'country1' in data and 'country2' in data and data['country1'] == data['country2']:
        while data['country1'] == data['country2']:
            data['country2'] = random.choice(ISO_3166['name'])


    return data

def populate_data(
        template_name: str,
        filled_entities: dict,
    ) -> dict:
    """
    Generate data for an explanation template.

    Parameters
    ----------
    filled_entities : dict
        The filled entities.

    Returns
    -------
    dict
        The data for the explanation template.
    """
    return DATA_GENERATORS[template_name](filled_entities)


def fill_explanation(
        explanation_template: dict,
        entities: dict,
        data: dict,
        detail: str = 'low',
    ) -> list:
    """
    Fill an explanation template with slot values.

    Parameters
    ----------
    explanation_template : dict
        The explanation template.
    entities : dict
        The slot values in the form {'property': "...", 'subject': "...", ...}.
    detail : int
        The level of detail to include in the explanation. Can be 'low', 'med', or 'high'.

    Returns
    -------
    dict
        The filled explanation template.
    """
    if detail not in ['low', 'med', 'high']:
        raise ValueError('detail must be one of "low", "med", or "high"')

    filled_template = []

    slots = {**entities, **data}

    for i, step in enumerate(explanation_template, 1):

        label = step['label']
        detail_variants = step['explanations'][detail]
        explanation = random.choice(detail_variants)
        filled_layer = {'step': i, 'label': label, 'explanation': explanation.format(**slots)}
        filled_template.append(filled_layer)

    return filled_template

def fill(args):

    q, e, t = load_template(args.template)
    entities = populate_entities(e)
    data = populate_data(args.template, entities)
    question = q.format(**entities, **data)
    filled_explanation = fill_explanation(t, entities, data)

    filled_template = {
            'template': args.template,
            'question': question,
            'explanation': filled_explanation,
    }



    return filled_template















if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fill explanation templates with subjects, properties, objects etc.')
    parser.add_argument('-t', '--template', required=True, type=str, help='The name of the explanation template.')
    parser.add_argument('-n', '--number', default=1, type=int, help='The number of questions to generate.')
    parser.add_argument('-s', '--save', type=str, help='Name of survey.')
    parser.add_argument('-p', '--print', action='store_true', help='Print the filled explanation template.')
    args = parser.parse_args()

    filled_template = fill(args)

    if args.print:
        # center filled_explanation['question'] in 80 equals signs
        print(f' {filled_template["question"]} '.center(80, '='))



        for step in filled_template['explanation']:
            # print step number with single leading zero
            print(f"Step {step['step']} ({step['label']}): {step['explanation']}")