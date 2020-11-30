import networkx as nx
from networkx.algorithms.operators import unary
import numpy as np


def tuple_argmax(tuple_list):
    argmax, max = tuple_list[0]
    for x in tuple_list:
        if x[1] > max:
            argmax = x[0]
            max = x[1]
    return argmax


def max_clique(graph):
    complement_graph = unary.complement(graph)
    while complement_graph.size() > 0:
        degrees = list(complement_graph.degree)
        max_node = tuple_argmax(degrees)
        complement_graph.remove_node(max_node)
    return list(unary.complement(complement_graph))
