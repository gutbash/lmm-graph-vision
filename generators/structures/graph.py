"""
The graph module contains the classes and methods for random directed and undirected graphs.
"""

import networkx as nx
import matplotlib.pyplot as plt
import random
from pathlib import Path
from typing import Optional, Literal

Color = Literal['#88d7fe', '#feaf88', '#eeeeee']
Shape = Literal['o', 's', 'd']
Font = Literal['sans-serif', 'serif', 'monospace']
Thickness = Literal['0.5', '1.0', '1.5']

class Graph:
    
    large: bool = False
    graph_skeleton: nx.DiGraph = None
    graph_filled: nx.DiGraph = None
    
    def adjacency_list(self) -> dict:
        """
        Returns the adjacency list of the graph with node values as keys
        
        Parameters
        ----------
        None
        
        Returns
        -------
        dict
            the adjacency list of the graph
        """
        
        adj_list = {}
        for node in self.graph_filled.nodes:
            # Get the value of the node
            node_value = self.graph_filled.nodes[node]['value']
            # Get the values of the neighbors
            neighbor_values = [self.graph_filled.nodes[neighbor]['value'] for neighbor in self.graph_filled.neighbors(node)]
            # Store in the dictionary
            adj_list[node_value] = neighbor_values

        return adj_list
    
    def breadth_first_search(self) -> list:
        """
        Returns the BFS traversal of the graph
        
        Parameters
        ----------
        None
        
        Returns
        -------
        list
            the BFS traversal of the graph
        """
        if not self.graph_filled:
            raise ValueError("The graph is empty.")
        start_node = next(iter(self.graph_filled))
        bfs_nodes = list(nx.bfs_tree(self.graph_filled, source=start_node).nodes)
        bfs_values = [self.graph_filled.nodes[node]['value'] for node in bfs_nodes]
        return bfs_values
    
    def depth_first_search(self) -> list:
        """
        Returns the DFS traversal of the graph
        
        Parameters
        ----------
        None
        
        Returns
        -------
        list
            the DFS traversal of the graph
        """
        if not self.graph_filled:
            raise ValueError("The graph is empty.")
        start_node = next(iter(self.graph_filled))
        dfs_nodes = list(nx.dfs_tree(self.graph_filled, source=start_node).nodes)
        dfs_values = [self.graph_filled.nodes[node]['value'] for node in dfs_nodes]
        return dfs_values

class UndirectedGraph(Graph):
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
    default_file_name : str
        the default file name for the image
    yaml_structure_type : str
        the YAML structure type
    formal_name : str
        the formal name of the structure
        
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
    
    default_file_name: str = 'ug_test.png'
    yaml_structure_type: str = 'undirected_graph'
    formal_name: str = 'Undirected Graph'
    
    def __init__(self, large: bool = False) -> None:
        """
        Constructs all the necessary attributes for an UndirectedGraph object
        
        Parameters
        ----------
        large : bool
            whether the graph should be large or not
        """
        self.large = large
        
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
            num_nodes = random.randint(3, 10)

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
        
        values = [i for i in range(1, len(self.graph_filled)+1)]

        for i, node in enumerate(self.graph_filled.nodes):
            self.graph_filled.nodes[node]['value'] = values[i]
        
    def draw(self, save: bool = False, path: Optional[Path] = None, show: bool = True, shape: Shape = 'o', color: Color = '#88d7fe', font: Font = 'sans-serif', thickness: Thickness = '1.0') -> None:
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
        shape : Shape
            the shape of the nodes (default is 'o')
        color : Color
            the color of the nodes (default is '#88d7fe' aka sky blue)
        font : Font
            the font of the node labels (default is 'sans-serif')
        thickness : Thickness
            the thickness of the edges (default is '1.0')

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
        
        labels = {node: self.graph_filled.nodes[node]['value'] for node in self.graph_filled.nodes}
        nx.draw(self.graph_filled, pos, with_labels=True, font_weight='bold', node_size=400, node_color=color, node_shape=shape, font_family=font, labels=labels, font_size=10, linewidths=float(thickness), width=1.0, alpha=1.0, edgecolors='black')

        if save:
            if path is None:
                path = "output.png"  # Default file name
            plt.savefig(fname=path, format='png', dpi=dpi)
            
        if show:
            plt.show()

class DirectedGraph(Graph):
    """
    A directed graph
    
    Attributes
    ----------
    large : bool
        whether the graph should be large or not
    default_file_name : str
        the default file name for the image
    yaml_structure_type : str
        the YAML structure type
    formal_name : str
        the formal name of the structure
        
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
    
    default_file_name: str = 'dg_test.png'
    yaml_structure_type: str = 'directed_graph'
    formal_name: str = 'Directed Graph'
    
    def __init__(self, large: bool = False) -> None:
        """
        Constructs all the necessary attributes for a DirectedGraph object
        
        Parameters
        ----------
        large : bool
            whether the graph should be large or not
        """
        self.large = large

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
            num_nodes = random.randint(3, 10)

        if num_nodes <= 0:
            return

        max_connections = 2
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
        
        values = [i for i in range(1, len(self.graph_filled)+1)]

        for i, node in enumerate(self.graph_filled.nodes):
            self.graph_filled.nodes[node]['value'] = values[i]

    def draw(self, save: bool = False, path: Optional[Path] = None, show: bool = True, shape: Shape = 'o', color: Color = '#88d7fe', font: Font = 'sans-serif', thickness: Thickness = '1.0') -> None:
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
        shape : Shape
            the shape of the nodes (default is 'o')
        color : Color
            the color of the nodes (default is '#88d7fe' aka sky blue)
        font : Font
            the font of the node labels (default is 'sans-serif')
        thickness : Thickness
            the thickness of the edges (default is '1.0')

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
        
        labels = {node: self.graph_filled.nodes[node]['value'] for node in self.graph_filled.nodes}
        nx.draw(self.graph_filled, pos, with_labels=True, font_weight='bold', node_size=400, node_color=color, node_shape=shape, font_family=font, labels=labels, font_size=10, linewidths=float(thickness), width=1.0, alpha=1.0, edgecolors='black')

        if save:
            if path is None:
                path = "output.png"  # Default file name
            plt.savefig(fname=path, format='png', dpi=dpi)
            
        if show:
            plt.show()
