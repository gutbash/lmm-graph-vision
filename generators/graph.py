"""
The graph module contains the classes for generating directed and undirected graphs.
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