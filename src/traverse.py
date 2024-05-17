"""
Class for searching and preparing an inference graph explanation
"""
import pickle
from itertools import groupby
from pathlib import Path

import networkx as nx
from frank.graph import InferenceGraph

from src import utils


class Traverse:
    """
    Class to describe a graph for preference learning experiment. Creates an
    exhaustive (i.e., including every edge) description of the graph.
    """
    def __init__(
            self,
            G: InferenceGraph,
            polish_nodes: bool = True,
            polish_edges: bool = True,
            remove_post_lookup: bool = True,
            create_agg: bool = True,
        ) -> None:
        """
        Initialize the class.

        Parameters
        ----------
        G : InferenceGraph
            The graph to traverse.
        polish_nodes : bool
            Rename nodes to integers.
        polish_edges : bool
            Rename edges and add data.
        remove_post_lookup : bool
            Remove nodes which are successors of lookup edges.
        create_agg : bool
            Create the aggregation graph.
        """
        # denote 'standard' graph by 'D' for 'Decomposition'
        self.D = G
        # rename node to int(node)
        if polish_nodes:
            self.polish_node_data()

        # replace 'Lookup' with 'lookup', add 'dir' and 'depth' to edge data
        if polish_edges:
            self.polish_edge_data()

        # remove nodes which are successors of lookup edges
        if remove_post_lookup:
            self.remove_post_lookup_nodes()

        if create_agg:
            self.A = self.create_agg_graph()
            self.G = nx.compose(self.D, self.A)
        # else:
        #     self.G = self.D

    def polish_node_data(
            self,
        ) -> None:
        """
        Rename nodes to integers.
        """
        H = InferenceGraph()
        H.add_nodes_from([(int(n), data) for n, data in self.D.nodes(data=True)])
        H.add_edges_from([(int(u), int(v), data) for u, v, data in self.D.edges(data=True)])

        self.D = H

    def polish_edge_data(
            self,
        ) -> None:
        """
        Rename 'Lookup' to 'lookup' for consistency. Also add 'dir' and 'depth'
        attributes, representing the direction of the edge and the depth of the
        decomposition.
        """
        for u, v, data in self.D.edges(data=True):
            if data['label'] == 'Lookup':
                self.D[u][v]['label'] = 'lookup'
            self.D[u][v]['dir'] = 'dec'
            self.D[u][v]['depth'] = self.D.alist(u).attributes['meta']['depth']

    def remove_post_lookup_nodes(
            self,
        ) -> None:
        """
        Remove nodes which are successors of lookup edges.
        """
        nodes_to_remove = []

        def _search_and_remove(node):
            """Recursive function to search and remove nodes."""
            for successor in self.D.successors(node):
                edge_data = self.D.get_edge_data(node, successor)
                if edge_data and edge_data['label'] not in {'lookup', 'comp_lookup'}:
                    nodes_to_remove.append(successor)
                    _search_and_remove(successor)

        for _, v, data in self.D.edges(data=True):
            if data['label'] in {'lookup', 'comp_lookup'}:
                _search_and_remove(v)

        for node in nodes_to_remove:
            self.D.remove_node(node)

    def create_agg_graph(
            self,
        ) -> InferenceGraph:
        """
        Add inference function as edge attribute to upward/aggregation pass.

        Returns
        -------
        A: InferenceGraph
            The graph with the inference function added.
        """
        A = InferenceGraph()

        for edge in self.D.edges():

            # swap edge direction
            u = edge[1]
            v = edge[0]

            # label from inference function of source node
            h = utils.format_alist(self.D.alist(u))['function']
            A.add_edge(u, v, label=h, dir='agg', depth=self.D.get_edge_data(v, u)['depth'])

        return A

    def traverse(
            self,
            direction: str,
        ) -> list:
        """
        Get order for node description.

        Parameters
        ----------
        direction : str
            The direction of the edges: 'dec' or 'agg'.
        """
        G = self.D.edges(data=True)

        # group edges by source node
        groups = groupby(G, lambda x: x[0])

        # collapse single-node groups into previous layer
        layers = [[]]
        for _, group in groups:
            group = list(group)
            if len(group) > 1:
                layers.append(list(group))
            elif len(group) == 1:
                layers[-1].extend(group)

        # group each layer by edge label
        labelled_layers = []
        for layer in layers:
            groups = groupby(layer, lambda x: x[2]['label'])
            labelled_layers.extend([list(g) for _, g in groups])

        # create final 'chunks'
        chunks = []
        for layer in labelled_layers:

            # simply reformat layer
            if direction == 'dec':
                label = layer[0][2]['label']
                edges = list(layer)

            # reverse edge and grab inference function from new source node
            elif direction == 'agg':
                # if duplicate labels, maybe cluster again on those?
                label = self.D.alist(layer[0][1]).attributes['h']
                edges = [(v, u, {**d, 'dir': 'agg', 'label': label}) for u, v, d in layer]

            chunks.extend([{label: edges}])

        if direction == 'agg':
            return chunks[::-1]

        return chunks

    def pretty_print_layers(
            self,
            detailed: bool = False,
        ) -> None:
        """
        Print the layers of the graph. Mostly for debugging/testing.

        Parameters
        ----------
        detailed : bool
            Print detailed information about each edge.
        """
        dec_layers, agg_layers = self.traverse('dec'), self.traverse('agg')

        print(" Graph Layers ".center(80, '='))
        for i, layer in enumerate([*dec_layers, *agg_layers]):
            print(f"layer {i}: len = {len(*layer.values())}")
            for label, tuples in layer.items():
                direction = tuples[0][2]['dir']
                if detailed:
                    for u, v, _ in tuples:
                        print(f"\t{u} -> {direction}:{label} -> {v}")
                else:
                    u, v, _ = zip(*tuples)
                    if len(tuples) == 1:
                        print(f"\t{u[0]} -> {direction}:{label} -> {v[0]}")
                    else:
                        min_u, max_u, min_v, max_v = min(u), max(u), min(v), max(v)
                        print(f"\t{min_u} ... {max_u} -> {direction}:{label} -> {min_v} ... {max_v}")


if __name__ == '__main__':

    path = Path('resources', 'frank_query_pickles', '152', '2023-12-07T12:31:28.929961.pkl')
    # path = Path('resources', 'frank_query_pickles', '152', '2023-12-06T16:22:41.291909.pkl')
    # path = Path('resources', 'frank_query_pickles', '151', '2023-12-01T12:42:30.681558.pkl')
    # path = Path('resources', 'frank_query_pickles', '151', '2023-12-01T12:42:23.554100.pkl')
    # path = Path('resources', 'frank_query_pickles', '151', '2023-12-01T12:09:46.203908.pkl')
    # path = Path('resources', 'frank_query_pickles', '2024-03-08T14:47:46.400794.pkl')
    path = Path('resources/frank_query_pickles/clean/B1/2024-03-08T14:47:46.400794.pkl')
    with path.open('rb') as f:
        q = pickle.load(f)

    graph = q['graph']

    T = Traverse(graph)
    T.pretty_print_layers(detailed=False)
