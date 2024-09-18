"""Functions for populating an entity dict with values."""
import json
import random
from pathlib import Path

import pandas as pd

with Path("resources/slot_values.json").open(encoding="utf-8") as f:
    SLOT_VALUES = json.load(f)

df = pd.read_csv('resources/iso-3166.csv')

def temporal_data(
        property: str = 'population',
    ) -> dict:
    """Create data for explanation templates."""
    data = {}

    if property == 'population':
        # create random time-value data for years starting in range 2008-2012 and ending 2021-2023
        t_min = random.randint(2008, 2011)
        t_max = random.randint(2021, 2023)
        steps = random.randint(8, 10)
        # steadily increasing values on order 1e8 with noise
        v_min = random.randint(31e6, 38e6)
        v_max = random.randint(41e6, 48e6)

        value = v_max + random.randint(1e6, 2e6)

        data['t_min'] = t_min
        data['t_max'] = t_max
        data['steps'] = steps
        data['v_min'] = v_min
        data['v_max'] = v_max
        data['value'] = value

    return data

def A1(
        filled_entities: list[str],
    ) -> dict:
    """Create data for explanation templates for pattern A2."""
    # create random time-value data for years starting in range 2008-2012 and ending 2021-2023
    future = random.randint(2026, 2030)
    t_min = random.randint(2008, 2011)
    t_max = random.randint(2021, 2023)
    steps = random.randint(8, 10)

    data = {}
    data['t_min'] = t_min
    data['t_max'] = t_max
    data['steps'] = steps
    data['future'] = future

    return data

def B1(
        filled_entities: list[str],
) -> dict:
    """Create data for explanation templates for pattern B1."""
    countries_in_region = df[df['sub-region'] == filled_entities['region']]['name']
    n_countries = len(countries_in_region)
    country_sample = '{}, {} and {}'.format(*random.sample(list(countries_in_region), 3))

    if filled_entities['property'] == 'population':
        if filled_entities['operator'] == 'lowest':
            value = random.randint(1e5, 4e5)
        elif filled_entities['operator'] == 'highest':
            value = random.randint(8e6, 12e6)

    data = {}
    data['country_sample'] = country_sample
    data['n_countries'] = n_countries
    data['answer'] = random.choice(list(countries_in_region))
    data['past'] = random.randint(2012, 2021)

    return data

def C1(
        filled_entities: list[str],
    ) -> dict:
    """Create data for explanation templates for pattern C1."""
    countries_in_region = df[df['sub-region'] == filled_entities['region']]['name']
    n_countries = len(countries_in_region)
    country_sample = '{}, {} and {}'.format(*random.sample(list(countries_in_region), 3))

    past = random.randint(2008, 2021)
    future = random.randint(2026, 2030)

    x = random.sample(list(countries_in_region), 1)[0]
    y = random.sample(list(countries_in_region), 1)[0]

    t_min = random.randint(2008, 2011)
    t_max = random.randint(2021, 2023)

    data = {}
    data['t_min'] = t_min
    data['t_max'] = t_max
    data['n_countries'] = n_countries
    data['country_sample'] = country_sample

    data['past'] = past
    data['future'] = future

    data['$x'] = x

    return data

def D1(
        filled_entities: list[str],
    ) -> dict:
    """Create data for explanation templates for pattern D1."""
    data = {}
    data['future'] = random.randint(2026, 2030)
    data['t_min1'] = random.randint(2008, 2011)
    data['t_max1'] = random.randint(2021, 2023)
    data['t_min2'] = random.randint(2008, 2011)
    data['t_max2'] = random.randint(2021, 2023)

    return data