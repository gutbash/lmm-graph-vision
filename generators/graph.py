"""
The graph module contains the classes and methods for generating directed and undirected graphs.
"""

import networkx as nx
import matplotlib.pyplot as plt
import random

class GraphNode:
    def __init__(self, value: int):
        self.value = value
        self.neighbors = []

class UndirectedGraph:
    def __init__(self, large: bool = False):
        self.large = large

class DirectedGraph:
    def __init__(self, large: bool = False):
        self.large = large
        
def generate_undirected_graph(large: bool = False, save: bool = False, path: str = 'ug_test.png', show: bool = False):
    pass

def generate_directed_graph(large: bool = False, save: bool = False, path: str = 'dg_test.png', show: bool = False):
    pass