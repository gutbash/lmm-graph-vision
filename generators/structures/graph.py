"""
The graph module contains the classes and methods for random directed and undirected graphs.
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
from pathlib import Path
from typing import Optional

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
    
    value: int
    neighbors: list
    
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
    graph_skeleton : nx.Graph
        the basic structure of the undirected graph
    graph_filled : nx.Graph
        the filled undirected graph
        
    Methods
    -------
    __init__(large: bool = False)
        Constructs all the necessary attributes for the UndirectedGraph object
    generate()
        Generates a random undirected graph
    fill()
        Fills the graph nodes with the given values
    draw(save: bool = False, path: Optional[Path] = None, show: bool = True)
        Visualizes the generated undirected graph
    """
    
    large: bool
    graph_skeleton: nx.Graph
    graph_filled: nx.Graph
    
    def __init__(self, large: bool = False) -> None:
        """
        Constructs all the necessary attributes for an UndirectedGraph object
        
        Parameters
        ----------
        large : bool
            whether the graph should be large or not
        """
        
        self.large = large
        self.graph_skeleton = None
        self.graph_filled = None
        
    def generate(self) -> None:
        """
        Generates a random undirected graph with basic structure
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        G = nx.Graph()

        if self.large:
            num_nodes = random.randint(11, 20)
        else:
            num_nodes = random.randint(1, 10)

        if num_nodes <= 0:
            return

        max_connections = 4
        connections_count = {node: 0 for node in range(num_nodes)}

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.choice([True, False]) and connections_count[i] < max_connections and connections_count[j] < max_connections:
                    G.add_edge(i, j)
                    connections_count[i] += 1
                    connections_count[j] += 1
                    
        self.graph_skeleton = G

    def fill(self) -> None:
        """
        Fills the graph nodes with the given values

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        self.graph_filled = self.graph_skeleton.copy()
        
        values = [random.randrange(1, 100, 1) for _ in range(len(self.graph_filled))]

        for i, node in enumerate(self.graph_filled.nodes):
            self.graph_filled.nodes[node]['value'] = values[i]
        
    def draw(self, save: bool = False, path: Optional[Path] = None, show: bool = True) -> None:
        """
        Visualizes the generated undirected graph
        
        Parameters
        ----------
        save : bool
            whether or not to save the image
        path : Optional[Path]
            the path to the image
        show : bool
            whether or not to show the image

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
        
        pos = nx.spring_layout(self.graph_filled, seed=42)  # Set seed for reproducibility
    
        # Create a figure with the calculated size
        plt.figure(figsize=(figure_size, figure_size))
        
        nx.draw(self.graph_filled, pos, width=1.57, with_labels=True, font_weight='bold', node_size=800, node_color='skyblue', labels=nx.get_node_attributes(self.graph_filled, 'value'))

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
    
    value: int
    out_neighbors: list
    in_neighbors: list
    
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
        
    Methods
    -------
    __init__(large: bool = False)
        Constructs all the necessary attributes for the DirectedGraph object
    generate()
        Generates a random directed graph
    fill()
        Fills the graph nodes with the given values
    draw(save: bool = False, path: Optional[Path] = None, show: bool = True)
        Visualizes the generated directed graph
    """
    
    large: bool
    graph_skeleton: nx.DiGraph
    graph_filled: nx.DiGraph
    
    def __init__(self, large: bool = False) -> None:
        """
        Constructs all the necessary attributes for a DirectedGraph object
        
        Parameters
        ----------
        large : bool
            whether the graph should be large or not
        """
        
        self.large = large
        self.graph_skeleton = None
        self.graph_filled = None

    def generate(self) -> None:
        """
        Generates a random directed graph with basic structure
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        G = nx.DiGraph()

        if self.large:
            num_nodes = random.randint(11, 20)
        else:
            num_nodes = random.randint(1, 10)

        if num_nodes <= 0:
            return

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
                        
        self.graph_skeleton = G
        
    def fill(self) -> None:
        """
        Fills the graph nodes with the given values

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        self.graph_filled = self.graph_skeleton.copy()
        
        values = [random.randrange(1, 100, 1) for _ in range(len(self.graph_filled))]

        for i, node in enumerate(self.graph_filled.nodes):
            self.graph_filled.nodes[node]['value'] = values[i]

    def draw(self, save: bool = False, path: Optional[Path] = None, show: bool = True) -> None:
        """
        Visualizes the generated directed graph
        
        Parameters
        ----------
        save : bool
            whether or not to save the image
        path : Optional[Path]
            the path to the image
        show : bool
            whether or not to show the image

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
        
        pos = nx.spring_layout(self.graph_filled, seed=42)  # Set seed for reproducibility
        
        # Create a figure with the calculated size
        plt.figure(figsize=(figure_size, figure_size))
        
        nx.draw(self.graph_filled, pos, width=1.57, with_labels=True, font_weight='bold', node_size=800, node_color='skyblue', labels=nx.get_node_attributes(self.graph_filled, 'value'))

        if save:
            if path is None:
                path = "output.png"  # Default file name
            plt.savefig(fname=path, format='png', dpi=dpi)
            
        if show:
            plt.show()
