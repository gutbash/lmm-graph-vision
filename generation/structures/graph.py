"""Contains classes for directed and undirected graphs."""

import networkx as nx
import matplotlib.pyplot as plt
import random
from pathlib import Path
from typing import Optional, Literal
from utils.colors import hex_to_rgb_float

Color = Literal['#abe0f9', '#fee4b3', '#eeeeee']
Shape = Literal['o', 's', 'd']
Font = Literal['sans-serif', 'serif', 'monospace']
Width = Literal['0.5', '1.0', '1.5']

class Graph:
    """
    A graph
    """
    
    methods: list = ['breadth_first_search', 'depth_first_search', 'adjacency_list']
    
    def adjacency_list(self, structure_instance) -> dict:
        """
        Returns the adjacency list of the graph with node values as keys and lists of neighbor values as values.
        
        Returns
        -------
        dict
            the adjacency list of the graph
        """
        
        adj_list = {}
        for node in structure_instance.graph.nodes:
            # Get the value of the node
            node_value = structure_instance.graph.nodes[node]['value']
            # Get the values of the neighbors
            neighbor_values = [structure_instance.graph.nodes[neighbor]['value'] for neighbor in structure_instance.graph.neighbors(node)]
            # Store in the dictionary
            adj_list[node_value] = neighbor_values

        return adj_list
    
    def breadth_first_search(self, structure_instance) -> list:
        """
        Returns the BFS traversal of the graph as a list of values.
        
        Returns
        -------
        list
            the BFS traversal of the graph
        """
        if not structure_instance.graph:
            raise ValueError("The graph is empty.")
        start_node = next(iter(structure_instance.graph))
        bfs_nodes = list(nx.bfs_tree(structure_instance.graph, source=start_node).nodes)
        bfs_values = [structure_instance.graph.nodes[node]['value'] for node in bfs_nodes]
        return bfs_values
    
    def depth_first_search(self, structure_instance) -> list:
        """
        Returns the DFS traversal of the graph as a list of values.
        
        Returns
        -------
        list
            the DFS traversal of the graph
        """
        if not structure_instance.graph:
            raise ValueError("The graph is empty.")
        start_node = next(iter(structure_instance.graph))
        dfs_nodes = list(nx.dfs_tree(structure_instance.graph, source=start_node).nodes)
        dfs_values = [structure_instance.graph.nodes[node]['value'] for node in dfs_nodes]
        return dfs_values

class UndirectedGraph(Graph):
    """
    An undirected graph
    
    Attributes
    ----------
    large : bool
        whether the graph should be large or not
    graph : nx.Graph
        the basic structure of the undirected graph
    default_file_name : str
        the default file name for the image
    yaml_structure_type : str
        the YAML structure type
    formal_name : str
        the formal name of the structure
    """
    large: bool = False
    graph: nx.DiGraph = nx.DiGraph()
    
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
        self.graph = nx.Graph()
        
    def generate(self, num_nodes: int = None, num_edges: int = None) -> None:
        """
        Generates a random undirected graph with basic structure.
        """
        G = nx.Graph()

        if not num_nodes:
            num_nodes = random.randint(3, 18)
            
        if num_nodes <= 0:
            return

        for i in range(1, num_nodes):
            G.add_edge(i - 1, i)

        if not num_edges:
            additional_edges = random.randint(1, num_nodes * (num_nodes - 1) // 4)
        else:
            additional_edges = num_edges - (num_nodes - 1)  # already added num_nodes-1 edges

        while additional_edges > 0:
            source, target = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
            if source != target and not G.has_edge(source, target):
                G.add_edge(source, target)
                additional_edges -= 1

        self.graph = G

    def fill(self) -> None:
        """
        Fills the graph nodes with given values.
        """
        values = [i for i in range(1, len(self.graph)+1)]

        for i, node in enumerate(self.graph.nodes):
            self.graph.nodes[node]['value'] = values[i]
        
    def draw(self, save: bool = False, path: Optional[Path] = None, show: bool = True, shape: Shape = 'o', color: Color = '#abe0f9', font: Font = 'sans-serif', width: Width = '1.0', resolution: int = 512, arrow_style: str = '-') -> None:
        """
        Visualizes the generated undirected graph.
        
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
            the color of the nodes (default is '#abe0f9' aka sky blue)
        font : Font
            the font of the node labels (default is 'sans-serif')
        width : Width
            the width of the edges (default is '1.0')
        
        Notes
        -----
        The visualization is done using the networkx and matplotlib libraries.
        
        The nodes are labeled with their values.
        
        The graph is displayed using matplotlib.
        """
        
        # DPI for the output
        dpi = 100

        # Calculate the figure size in inches for a 512x512 pixel image
        figure_size = resolution / dpi  # 5.12 when dpi is 100
        scale_factor = figure_size / (512 / dpi)

        # Adjusted sizes
        node_size = 800 * (scale_factor ** 2)
        font_size = 12 * scale_factor
        edge_width = float(width) * scale_factor
        
        pos = nx.spring_layout(self.graph, seed=42)  # Set seed for reproducibility
    
        # Create a figure with the calculated size
        plt.figure(figsize=(figure_size, figure_size))
        
        labels = {node: self.graph.nodes[node].get('value', node) for node in self.graph.nodes}
        nx.draw(self.graph, pos, with_labels=True, font_weight='bold', node_size=node_size, node_color=color, node_shape=shape, font_family=font, labels=labels, font_size=font_size, linewidths=edge_width, width=edge_width, edgecolors=hex_to_rgb_float(color, -50), alpha=1.0)

        if save:
            plt.savefig(fname=path if path else self.default_file_name, format='png', dpi=dpi)
        if show:
            plt.show()
            
        plt.close()

class DirectedGraph(Graph):
    """
    A directed graph
    
    Attributes
    ----------
    large : bool
        whether the graph should be large or not
    graph : nx.DiGraph
        the basic structure of the directed graph
    default_file_name : str
        the default file name for the image
    yaml_structure_type : str
        the YAML structure type
    formal_name : str
        the formal name of the structure
    """
    large: bool = False
    graph: nx.DiGraph = nx.DiGraph()
    
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
        self.graph = nx.DiGraph()

    def generate(self, num_nodes: int = None, num_edges: int = None) -> None:
        """
        Generates a directed graph with basic structure.
        """
        G = nx.DiGraph()

        if not num_nodes:
            num_nodes = random.randint(3, 18)
            
        if num_nodes <= 0:
            return

        # Create a connected graph (spanning tree)
        for i in range(1, num_nodes):
            G.add_edge(i - 1, i)

        # ~~Randomize additional edges with control to avoid clutter~~
        if not num_edges:
            additional_edges = random.randint(1, num_nodes * (num_nodes - 1) // 4)
        else:
            additional_edges = num_edges - (num_nodes - 1)  # Already added num_nodes-1 edges

        while additional_edges > 0:
            source, target = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
            if source != target and not G.has_edge(source, target):
                G.add_edge(source, target)
                additional_edges -= 1

        self.graph = G
        
    def fill(self) -> None:
        """
        Fills the graph nodes with the given values.
        """
        values = [i for i in range(1, len(self.graph)+1)]

        for i, node in enumerate(self.graph.nodes):
            self.graph.nodes[node]['value'] = values[i]

    def draw(self, save: bool = False, path: Optional[Path] = None, show: bool = True, shape: Shape = 'o', color: Color = '#abe0f9', font: Font = 'sans-serif', width: Width = '1.0', resolution: int = 512, arrow_style: str = '-|>') -> None:
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
            the color of the nodes (default is '#abe0f9' aka sky blue)
        font : Font
            the font of the node labels (default is 'sans-serif')
        width : Width
            the width of the edges (default is '1.0')
        
        Notes
        -----
        The visualization is done using the networkx and matplotlib libraries.
        
        The nodes are labeled with their values.
        
        The graph is displayed using matplotlib.
        """
        if self.graph is None or len(self.graph.nodes()) == 0:
            raise ValueError("Graph is empty or not generated")
        
        pos = nx.spring_layout(self.graph, seed=42)  # Set seed for reproducibility

        # DPI for the output
        dpi = 100

        # Calculate the figure size in inches for a 512x512 pixel image
        figure_size = resolution / dpi  # 5.12 when dpi is 100
        scale_factor = figure_size / (512 / dpi)

        # Adjusted sizes
        arrow_size = 20 * scale_factor
        node_size = 800 * (scale_factor ** 2)
        font_size = 12 * scale_factor
        edge_width = float(width) * scale_factor
        
        # Create a figure with the calculated size
        plt.figure(figsize=(figure_size, figure_size))
        
        labels = {node: self.graph.nodes[node].get('value', node) for node in self.graph.nodes}
        nx.draw(self.graph, pos, with_labels=True, font_weight='bold', arrowsize=arrow_size, node_size=node_size, node_color=color, node_shape=shape, font_family=font, labels=labels, font_size=font_size, linewidths=edge_width, width=edge_width, alpha=1.0, edgecolors=hex_to_rgb_float(color, -50), arrowstyle=arrow_style)

        if save:
            plt.savefig(fname=path if path else self.default_file_name, format='png', dpi=dpi)
        if show:
            plt.show()
        
        plt.close()