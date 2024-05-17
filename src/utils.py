"""
Utils for doing graph stuff
"""
import math

from frank.alist import Alist
from frank.graph import InferenceGraph


def format_alist(
        alist: Alist,
        digits: int = 3,
        if_not_replace: str = 'use_old'
    ) -> dict:
    """Make an alist's keys more verbose and restructure certain keys and
    values. Return a new dict with the new keys and values.

    Parameters
    ----------
    alist : Alist
        Alist to format.
    digits : int
        Number of significant digits to round to.
    if_not_replace : str
        What to do if the value is not in the replace dict. Options are:
        'use_old' (default), 'delete'.

    Returns
    -------
    dict
        Alist with formatted keys and values.
    """
    alist_attrs = alist.attributes
    replace = {
        'h': 'function',
        's': 'subject',
        'p': 'property',
        'o': 'object',
        't': 'time',
        'v': 'value',
        'u': 'uncertainty',
        'fp': 'parameters',
        'kb': 'knowledge_base',
        'meta': 'meta',
        'data_sources': 'data_sources',
        'worldbank': 'World Bank Open Data',
        'wikidata': 'Wikidata',
        '?y0': '?y0',
        'min': 'minimum',
        'max': 'maximum',
    }

    # Replace instantiation variable
    # if str(alist_attrs['o']).startswith('?y'):
    #     alist_attrs['o'] = alist_attrs[alist_attrs['o']]

    def _recursive_replace(
            alist: dict,
        ) -> dict:
        """Recursively replace a dictionary's keys with keys in replace"""
        sub_alist = {}
        for k, v in alist.items():

            # delete key if v = ''
            if v == '':
                continue

            # delete key if in list
            if k in ['cx', 'branch_type', 'state', 'cost', 'node_type', 'what', 'why', 'how', 'where', 'when', 'how']:
                continue

            # Recursively replace keys in nested dicts
            if isinstance(v, dict):
                v = _recursive_replace(v)

            # Recursively replace keys in nested lists
            if isinstance(v, list):
                for i, x in enumerate(v):
                    if isinstance(x, dict):
                        v = _recursive_replace(x)

                    # Replace data source names
                    if x in replace:
                        alist[k][i] = replace[x]

            # Replace keys
            if k in replace:
                if isinstance(v, (int, float)):
                    sub_alist[replace[k]] = str(v)
                else:
                    sub_alist[replace[k]] = v
            else:
                if if_not_replace == 'delete':
                    continue
                sub_alist[k] = v

        return sub_alist

    new_alist = _recursive_replace(alist_attrs)

    for k in ['object', 'value', 'uncertainty', '?y0']:
        try:
            new_alist[k] = round_to_sf(float(new_alist[k]), digits)

        except (ValueError, KeyError):
            pass

    return new_alist

def round_to_sf(
        number: float | int,
        sf: int = 3,
        as_str: bool = True,
    ) -> float | str:
    """Round a number to a given number of significant figures.

    Parameters
    ----------
    number : float | int
        The number to round.
    sf : int
        The number of significant figures to round to.
    as_str : bool
        Whether to return the number as a string or float.

    Returns
    -------
    out: float | str
        The rounded number.
    """
    rounded = round(number, -int(math.floor(math.log10(abs(number))) - (sf - 1)))
    if str(rounded).rsplit('.', maxsplit=1)[-1] == '0':
        rounded = int(rounded)
    out = format(rounded, ',') if as_str else rounded

    return out

def incoming_edge_label(
        G: InferenceGraph,
        node: int,
        direction: str,
    ) -> str:
    """
    Get the data of the incoming edge to a node.

    Parameters
    ----------
    G : InferenceGraph
        The graph to get the edge data from.
    node : int
        The node to get the incoming edge data of.
    direction: str
        The direction of the edge: 'agg' or 'dec'.

    Returns
    -------
    inc: str
        The label of the incoming edge.
    """
    all_inc = [(p, node, G.get_edge_data(p, node)) for p in G.predecessors(node)]

    # handle leaf nodes where edge direction inflects
    if set(d['dir'] for _, _, d in all_inc) == {'dec'} and direction == 'agg':

        # just take label of only incoming edge
        inc = all_inc[0][2]['label']

    else:
        inc = next((d['label'] for _, v, d in all_inc if d['dir'] == direction), None)

    return inc


def outgoing_edge_label(
        G: InferenceGraph,
        node: str,
        direction: str,
    ) -> str:
    """
    Get the data of the outgoing edge to a node.

    Parameters
    ----------
    node : str
        The node to get the outgoing edge data of.
    direction: str
        The direction of the edge: 'agg' or 'dec'.

    Returns
    -------
    out: str
        The label of the outgoing edge.
    """
    all_out = [(p, node, G.get_edge_data(p, node)) for p in G.successors(node)]
    out = next((d['label'] for _, v, d in all_out if d['dir'] == direction), None)

    return out

def get_data_sources(
        nodes: dict | list[dict]
    ) -> list:
    """
    Get the data sources of a node.

    Parameters
    ----------
    nodes : str | list[str]
        The node(s) to get the data sources of.

    Returns
    -------
    list
        The data sources of the node.
    """
    if isinstance(nodes, dict):
        nodes = [nodes]
    n = [node['meta']['data_sources'] for node in nodes]
    data_sources = set()
    for x in n:
        data_sources.update(x)

    return ' and '.join(list(data_sources))

def clean_alist(
        alist: Alist | dict,
        valid_keys: list[str],
    ) -> dict:
    """
    Remove keys from an alist that are not in valid_keys.
    """
    if isinstance(alist, Alist):
        alist = alist.attributes

    return {k: v for k, v in alist.items() if k in valid_keys}
