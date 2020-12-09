import networkx as nx
from networkx.algorithms.operators import unary
import numpy as np
from math import inf


def tuple_argmax(tuple_list):
    argmax, max_l = tuple_list[0]
    for x in tuple_list:
        if x[1] > max_l:
            argmax = x[0]
            max_l = x[1]
    return argmax


def tuple_argmin_gmax(tuple_list, weights):
    argmin, min_l = tuple_list[0][0], inf
    ind = 0
    for i in range(len(tuple_list)):
        x = tuple_list[i]
        if x[1] > 0 and weights[i]/(x[1]*(x[1]+1)) < min_l:
            argmin = x[0]
            min_l = weights[i]/(x[1]*(x[1]+1))
            ind = i
    return argmin, ind


def max_clique(graph):
    complement_graph = unary.complement(graph)
    while complement_graph.size() > 0:
        degrees = list(complement_graph.degree)
        max_node = tuple_argmax(degrees)
        complement_graph.remove_node(max_node)
    return list(unary.complement(complement_graph))


def max_weight_clique(graph, weights):
    complement_graph = unary.complement(graph)
    while complement_graph.size() > 0:
        degrees = list(complement_graph.degree)
        min_node, ind = tuple_argmin_gmax(degrees, weights)
        complement_graph.remove_node(min_node)
        weights.pop(ind)
    return list(unary.complement(complement_graph))
