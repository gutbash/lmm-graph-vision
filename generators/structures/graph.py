"""
The graph module contains the classes and methods for random directed and undirected graphs.
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
from pathlib import Path

class UndirectedGraphNode:
    """
    A node in an undirected graph
    
    Attributes
    ----------
    value : int
        the value of the node
    neighbors : list
        the neighbors of the node
        
    Methods
    -------
    __init__(value: int)
        Constructs all the necessary attributes for the UndirectedGraphNode object
    """
    
    def __init__(self, value: int) -> None:
        """
        Constructs all the necessary attributes for an UndirectedGraphNode object
        
        Parameters
        ----------
        value : int
            the value of the node
        """
        
        self.value = value
        self.neighbors = []

class UndirectedGraph:
    """
    An undirected graph
    
    Attributes
    ----------
    large : bool
        whether the graph should be large or not
    graph_nodes : list
        the nodes of the undirected graph
        
    Methods
    -------
    __init__(large: bool = False)
        Constructs all the necessary attributes for the UndirectedGraph object
    generate()
        Generates a random undirected graph
    draw(save: bool = False, path: Path = None, show: bool = True)
        Visualizes the generated undirected graph
    """
    
    def __init__(self, large: bool = False) -> None:
        """
        Constructs all the necessary attributes for an UndirectedGraph object
        
        Parameters
        ----------
        large : bool
            whether the graph should be large or not
        """
        
        self.large = large
        self.graph_nodes = []
        
    def generate(self):
        """
        Generates a random undirected graph with basic structure

        Returns
        -------
        nx.Graph
            the basic structure of the undirected graph
        """
        G = nx.Graph()

        if self.large:
            num_nodes = random.randint(11, 20)
        else:
            num_nodes = random.randint(1, 10)

        if num_nodes <= 0:
            return G

        max_connections = 4
        connections_count = {node: 0 for node in range(num_nodes)}

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.choice([True, False]) and connections_count[i] < max_connections and connections_count[j] < max_connections:
                    G.add_edge(i, j)
                    connections_count[i] += 1
                    connections_count[j] += 1

        return G

    def fill_graph(self, graph):
        """
        Fills the graph nodes with the given values

        Parameters
        ----------
        graph : nx.Graph
            the graph to be filled

        Returns
        -------
        nx.Graph
            the filled graph
        """
        values = [random.randrange(1, 100, 1) for _ in range(len(graph))]

        for i, node in enumerate(graph.nodes):
            graph.nodes[node]['value'] = values[i]

        return graph

    def visualize_graph(self, graph, with_labels=True):
        """
        Visualizes the undirected graph

        Parameters
        ----------
        graph : nx.Graph
            the graph to be visualized
        with_labels : bool, optional
            whether to display node labels, by default True

        Returns
        -------
        None
        """
        
    def draw(self, graph, save: bool = False, path: Path = None, show: bool = True, with_labels: bool = False) -> None:
        """
        Visualizes the generated undirected graph

        Returns
        -------
        None
        
        Notes
        -----
        The visualization is done using the networkx and matplotlib libraries.
        
        The nodes are labeled with their values.

        The graph is drawn using the spring layout.
        
        The graph is displayed using matplotlib.
        """
        
        # DPI for the output
        dpi = 100
        
        # Calculate the figure size in inches for a 512x512 pixel image
        figure_size = 512 / dpi  # 5.12 when dpi is 100
        
        pos = nx.spring_layout(graph, seed=42)  # Set seed for reproducibility
    
        # Create a figure with the calculated size
        plt.figure(figsize=(figure_size, figure_size))
        
        nx.draw(graph, pos, width=1.57, with_labels= with_labels, font_weight='bold', node_size=800, node_color='skyblue', labels=nx.get_node_attributes(graph, 'value'))

        if save:
            if path is None:
                path = "output.png"  # Default file name
            plt.savefig(fname=path, format='png', dpi=dpi)
            
        if show:
            plt.show()

class DirectedGraphNode:
    """
    A node in a directed graph
    
    Attributes
    ----------
    value : int
        the value of the node
    out_neighbors : list
        the out neighbors of the node
    in_neighbors : list
        the in neighbors of the node
        
    Methods
    -------
    __init__(value: int)
        Constructs all the necessary attributes for the DirectedGraphNode object
    """
    
    def __init__(self, value: int) -> None:
        """
        Constructs all the necessary attributes for a DirectedGraphNode object
        
        Parameters
        ----------
        value : int
            the value of the node
        """
        
        self.value = value
        self.out_neighbors = []
        self.in_neighbors = []

class DirectedGraph:
    """
    A directed graph
    
    Attributes
    ----------
    large : bool
        whether the graph should be large or not
    graph_nodes : list
        the nodes of the directed graph
        
    Methods
    -------
    __init__(large: bool = False)
        Constructs all the necessary attributes for the DirectedGraph object
    generate()
        Generates a random directed graph
    draw(save: bool = False, path: Path = None, show: bool = True)
        Visualizes the generated directed graph
    """
    
    def __init__(self, large: bool = False) -> None:
        """
        Constructs all the necessary attributes for a DirectedGraph object
        
        Parameters
        ----------
        large : bool
            whether the graph should be large or not
        """
        
        self.large = large
        self.graph_nodes = []

    def generate(self):
        """
        Generates a random directed graph with basic structure

        Returns
        -------
        nx.DiGraph
            the basic structure of the directed graph
        """
        G = nx.DiGraph()

        if self.large:
            num_nodes = random.randint(11, 20)
        else:
            num_nodes = random.randint(1, 10)

        if num_nodes <= 0:
            return G

        max_connections = 4
        out_connections_count = {node: 0 for node in range(num_nodes)}
        in_connections_count = {node: 0 for node in range(num_nodes)}

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.choice([True, False]):
                    if out_connections_count[i] < max_connections and in_connections_count[j] < max_connections:
                        G.add_edge(i, j)
                        out_connections_count[i] += 1
                        in_connections_count[j] += 1

        return G
        
    def fill_graph(self, graph):
        """
        Fills the graph nodes with the given values

        Parameters
        ----------
        graph : nx.DiGraph
            the graph to be filled

        Returns
        -------
        nx.DiGraph
            the filled graph
        """
        values = [random.randrange(1, 100, 1) for _ in range(len(graph))]

        for i, node in enumerate(graph.nodes):
            graph.nodes[node]['value'] = values[i]

        return graph

    def draw(self, graph, save: bool = False, path: Path = None, show: bool = True, with_labels: bool = False) -> None:
        """
        Visualizes the generated directed graph

        Returns
        -------
        None
        
        Notes
        -----
        The visualization is done using the networkx and matplotlib libraries.
        
        The nodes are labeled with their values.
        
        The graph is drawn using the spring layout.
        
        The graph is displayed using matplotlib.
        """
        
        # DPI for the output
        dpi = 100
        
        # Calculate the figure size in inches for a 512x512 pixel image
        figure_size = 512 / dpi  # 5.12 when dpi is 100
        
        pos = nx.spring_layout(graph, seed=42)  # Set seed for reproducibility
        
        # Create a figure with the calculated size
        plt.figure(figsize=(figure_size, figure_size))
        
        nx.draw(graph, pos, width=1.57, with_labels=with_labels, font_weight='bold', node_size=800, node_color='skyblue', labels=nx.get_node_attributes(graph, 'value'))

        if save:
            if path is None:
                path = "output.png"  # Default file name
            plt.savefig(fname=path, format='png', dpi=dpi)
            
        if show:
            plt.show()
