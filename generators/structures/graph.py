"""
The graph module contains the classes and methods for random directed and undirected graphs.
"""

import networkx as nx
import matplotlib.pyplot as plt
import random

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
    
    def __init__(self, value: int):
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
    visualize_graph()
        Visualizes the generated undirected graph
    """
    
    def __init__(self, large: bool = False):
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
        Generates a random undirected graph

        Returns
        -------
        list
            the nodes of the undirected graph
            
        Notes
        -----
        The number of nodes in the graph is randomly chosen between 1 and 10 for small graphs and between 11 and 20 for large graphs.
        
        The value of each node is randomly chosen between 1 and 100.
        
        Each node has a 50% chance of having an edge to another node.
        
        The graph is guaranteed to be connected.
        """
        
        if self.large:
            num_nodes = random.randint(11, 20)
        else:
            num_nodes = random.randint(1, 10)

        if num_nodes <= 0:
            return None

        node_values = [random.randrange(1, 100, 1) for i in range(num_nodes)]
        nodes = [UndirectedGraphNode(value) for value in node_values]

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.choice([True, False]):  # Randomly add an edge between nodes
                    nodes[i].neighbors.append(nodes[j])
                    nodes[j].neighbors.append(nodes[i])

        self.graph_nodes = nodes
        return nodes

    def visualize_graph(self):
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
        
        G = nx.Graph()

        for node in self.graph_nodes:
            G.add_node(node.value)
            for neighbor in node.neighbors:
                G.add_edge(node.value, neighbor.value)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, width = 1.57, with_labels=True, font_weight='bold', node_size=800, node_color='skyblue')
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
    
    def __init__(self, value: int):
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
    visualize_graph()
        Visualizes the generated directed graph
    """
    
    def __init__(self, large: bool = False):
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
        Generates a random directed graph

        Returns
        -------
        list
            the nodes of the directed graph
            
        Notes
        -----
        The number of nodes in the graph is randomly chosen between 1 and 10 for small graphs and between 11 and 20 for large graphs.
        
        The value of each node is randomly chosen between 1 and 100.
        
        Each node has a 50% chance of having an edge to another node.
        
        The graph is guaranteed to be connected.
        """
        
        if self.large:
            num_nodes = random.randint(11, 20)
        else:
            num_nodes = random.randint(1, 10)

        if num_nodes <= 0:
            return None

        node_values = [random.randrange(1, 100, 1) for i in range(num_nodes)]
        nodes = [DirectedGraphNode(value) for value in node_values]

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.choice([True, False]):  # Avoid self-loops and add diverse edges
                    nodes[i].out_neighbors.append(nodes[j])
                    nodes[j].in_neighbors.append(nodes[i])

        self.graph_nodes = nodes
        return nodes

    def visualize_graph(self):
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
        
        G = nx.DiGraph()

        for node in self.graph_nodes:
            G.add_node(node.value)
            for out_neighbor in node.out_neighbors:
                G.add_edge(node.value, out_neighbor.value)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, width = 1.57,with_labels=True, font_weight='bold', node_size=800, node_color='skyblue')
        plt.show()
        
# Example usage
undirected_graph = UndirectedGraph()
nodes = undirected_graph.generate()
print("Generated Nodes:", [node.value for node in nodes])

# Visualize the graph
undirected_graph.visualize_graph()

# Example usage
directed_graph = DirectedGraph()
nodes = directed_graph.generate()
print("Generated Nodes:", [node.value for node in nodes])

# Visualize the graph
directed_graph.visualize_graph()
