"""Contains classes for binary trees and binary search trees."""

import networkx as nx
import matplotlib.pyplot as plt
import random
from pathlib import Path
from typing import Optional, Literal, Any
from utils.colors import hex_to_rgb_float

Color = Literal['#abe0f9', '#fee4b3', '#eeeeee']
Shape = Literal['o', 's', 'd']
Font = Literal['sans-serif', 'serif', 'monospace']
Width = Literal['0.5', '1.0', '1.5']

Traversal = Literal['preorder', 'inorder', 'postorder']

class Tree:
    """
    A class used to represent a tree
    """
    
    methods: list = ['pre_order', 'in_order', 'post_order']
        
    def pre_order(self, structure_instance) -> list:
        """
        Traverses the tree in pre-order
        
        Returns
        -------
        list
            the pre-order traversal of the tree
        """
        def _pre_order(node: 'Tree.TreeNode') -> list:
            if node is None:
                return []
            return [structure_instance.graph.nodes[node.value]['value']] + _pre_order(node.left) + _pre_order(node.right)
        
        return _pre_order(structure_instance.root)
    
    def in_order(self, structure_instance) -> list:
        """
        Traverses the tree in in-order
        
        Returns
        -------
        list
            the in-order traversal of the tree
        """
        def _in_order(node: 'Tree.TreeNode') -> list:
            if node is None:
                return []
            return _in_order(node.left) + [structure_instance.graph.nodes[node.value]['value']] + _in_order(node.right)
        
        return _in_order(structure_instance.root)
        
    def post_order(self, structure_instance) -> list:
        """
        Traverses the tree in post-order
        
        Returns
        -------
        list
            the post-order traversal of the tree
        """
        def _post_order(node: 'Tree.TreeNode', graph: nx.Graph) -> list:
            if node is None:
                return []
            return _post_order(node.left, graph) + _post_order(node.right, graph) + [graph.nodes[node.value]['value']]

        return _post_order(structure_instance.root, structure_instance.graph)
    
    class TreeNode:
        """
        A class used to represent a tree node

        Attributes
        ----------
        value : int
            the value of the node
        left : Tree.TreeNode
            the left child of the node
        right : Tree.TreeNode
            the right child of the node
        """
        
        value: Optional[int]
        left: Optional['Tree.TreeNode']
        right: Optional['Tree.TreeNode']
        
        def __init__(self, value: Optional[int] = None):
            """
            Initializes the TreeNode object
            
            Parameters
            ----------
            value : int
                the value of the node (default is None)
            """
            self.value = value  # Node value
            self.left = None  # Left child
            self.right = None  # Right child

# Generate a random binary tree
class BinaryTree(Tree):
    """
    Represents a binary tree

    Attributes
    ----------
    large : bool
        whether to generate a large tree or not
    root : TreeNode
        the root of the binary tree
    pos : dict
        a dictionary of positions of nodes
    graph : nx.Graph
        the graph of the tree
    default_file_name : str
        the default file name for the image
    yaml_structure_type : str
        the structure type for the YAML file
    formal_name : str
        the formal name of the structure
    """
    large: bool = False
    root: Optional['Tree.TreeNode'] = None
    pos: dict = {}
    graph: Optional[nx.Graph] = nx.Graph()
    
    default_file_name: str = 'bt_test.png'
    yaml_structure_type: str = 'binary_tree'
    formal_name: str = 'Binary Tree'
    
    def __init__(self, large: bool = False) -> None:
        """
        Initializes the BinaryTree object
        
        Parameters
        ----------
        large : bool
            whether to generate a large tree with 11-20 nodes instead of 1-10 nodes or not (default is False)
        """
        self.large = large
        self.root = None
        self.pos = {}
        self.graph = nx.Graph()

    def generate(self, num_nodes: int = None) -> None:
        """
        Generates a random binary tree
        """
        if not num_nodes:
            num_nodes = random.randint(3, 18)

        if num_nodes <= 0:
            return

        root_value = random.randint(1, 99)
        root = Tree.TreeNode(root_value)
        values_set = {root_value}
        nodes = [root]
        self.root = root

        while len(nodes) < num_nodes:
            node = random.choice(nodes)

            if random.choice([True, False]):
                if node.left is None:
                    child_value = random.randint(1, 99)
                    while child_value in values_set:
                        child_value = random.randint(1, 99)

                    child = Tree.TreeNode(child_value)
                    node.left = child
                    values_set.add(child_value)
                    nodes.append(child)
            else:
                if node.right is None:
                    child_value = random.randint(1, 99)
                    while child_value in values_set:
                        child_value = random.randint(1, 99)

                    child = Tree.TreeNode(child_value)
                    node.right = child
                    values_set.add(child_value)
                    nodes.append(child)
            
        self._graphize(self.graph, self.root)
        
    def _graphize(self, T: nx.Graph, node: Tree.TreeNode, x: int = 0, y: int = 0, layer_height: Optional[int] = None, layer_width: Optional[int] = None) -> None:
        """
        Graphizes the binary tree

        Parameters
        ----------
        T : nx.Graph
            the graph to be drawn
        node : Tree.TreeNode
            the current node
        x : int
            the x coordinate of the current node (default is 0)
        y : int
            the y coordinate of the current node (default is 0)
        layer_height : Optional[int]
            the height of the current layer (default is None)
        layer_width : Optional[int]
            the width of the current layer (default is None)

        Raises
        ------
        ValueError
            if the node is None
        """
        
        if node:
            if layer_height is None:
                layer_height = random.randint(3, 6)  # Random height for each layer
            if layer_width is None:
                layer_width = 2

            current_pos = (x, y)
            self.pos[node.value] = current_pos

            if node.left:
                T.add_edge(node.value, node.left.value)
                self._graphize(T, node.left, x - layer_width, y - 1, layer_height / 2, layer_width / 2)
            if node.right:
                T.add_edge(node.value, node.right.value)
                self._graphize(T, node.right, x + layer_width, y - 1, layer_height / 2, layer_width / 2)
            else:
                T.add_node(node.value)
        else:
            raise ValueError("The node is None")
        self.graph = T

    def fill(self) -> None:
        """
        Fills the graph nodes with the given values
        """

        values = [random.randrange(1, 99, 1) for _ in range(len(self.graph))]
        #print("These are new node values:",values)
        for i, node in enumerate(self.graph.nodes):
            self.graph.nodes[node]['value'] = values[i]

    def draw(self, save: bool = False, path: Optional[Path] = None, show: bool = True, shape: Shape = 'o', color: Color = '#abe0f9', font: Font = 'sans-serif', width: Width = '1.0', resolution: int = 512, arrow_style: str = '-') -> None:
        """
        Draw the binary tree

        Parameters
        ----------
        save : bool
            whether to save the image or not (default is False)
        path : Optional[Path]
            the path to save the image (default is None)
        show : bool
            whether to show the image or not (default is True)
        shape : Shape
            the shape of the nodes (default is 'o')
        color : Color
            the color of the nodes (default is '#abe0f9' aka sky blue)
        font : Font
            the font of the labels (default is 'sans-serif')
        width : Width
            the width of the edges (default is '1.0')

        Raises
        ------
        ValueError
            if the root is None
            
        Notes
        -----
        The visualization is done using the networkx and matplotlib libraries.
        
        The nodes are labeled with their values.
        """
        
        if self.root is None:
            raise ValueError("The root is None")

        # DPI for the output
        dpi = 100

        # Calculate the figure size in inches for a 512x512 pixel image
        figure_size = resolution / dpi  # 5.12 when dpi is 100
        scale_factor = figure_size / (512 / dpi)

        # Adjusted sizes
        node_size = 800 * (scale_factor ** 2)
        font_size = 12 * scale_factor
        edge_width = float(width) * scale_factor

        # Create a figure with the calculated size
        plt.figure(figsize=(figure_size, figure_size))
        #print(self.pos)

        # Draw nodes and edges
        nx.draw(self.graph, self.pos, labels={node: data['value'] for node, data in self.graph.nodes(data=True)}, with_labels=True, font_weight='bold', node_size=node_size, node_color=color, node_shape=shape, font_family=font, font_size=font_size, linewidths=edge_width, width=edge_width, alpha=1.0, edgecolors=hex_to_rgb_float(color, -50))

        if save:
            plt.savefig(fname=path if path else self.default_file_name, format='png', dpi=dpi)
        if show:
            plt.show()   
        
        plt.close()
        
class BinarySearchTree(Tree):
    """
    Represents a binary search tree
    
    Attributes
    ----------
    large : bool
        whether to generate a large tree or not
    root : TreeNode
        the root of the binary search tree
    pos : dict
        a dictionary of positions of nodes
    graph : nx.Graph
        the graph of the tree
    default_file_name : str
        the default file name for the image
    yaml_structure_type : str
        the structure type for the YAML file
    formal_name : str
        the formal name of the structure
    """
    
    large: bool = False
    root: Optional['Tree.TreeNode'] = None
    pos: dict = {}
    graph: Optional[nx.Graph] = nx.Graph()
    
    default_file_name: str = 'bst_test.png'
    yaml_structure_type: str = 'binary_search_tree'
    formal_name: str = 'Binary Search Tree'
    
    def __init__(self, large: bool = False) -> None:
        """
        Initializes the BinarySearchTree object
        
        Parameters
        ----------
        large : bool
            whether to generate a large tree with 11-20 nodes instead of 1-10 nodes or not (default is False)
        """
        self.large = large
        self.root = None
        self.pos = {}
        self.graph = nx.Graph()

    def generate(self, num_nodes: int = None) -> None:
        """
        Generates a random binary search tree
        """
        if not num_nodes:
            num_nodes = random.randint(3, 18)

        if num_nodes <= 0:
            return

        # Generate unique values for the nodes
        values = random.sample(range(1, 100), num_nodes)

        # Insert nodes with random values while maintaining the BST property
        self.root = None
        for value in values:
            self.insert(value)

        self._graphize(self.graph, self.root)

    def insert(self, value: int) -> None:
        """
        Inserts a new node with the given value into the binary search tree
        """
        new_node = Tree.TreeNode(value)

        if self.root is None:
            self.root = new_node
            #print("This is the root:",self.root.value)
        else:
            current = self.root
            while True:
                if value < current.value:
                    if current.left is None:
                        current.left = new_node
                        #print("This is the left:",current.left.value)
                        break
                    current = current.left
                else:
                    if current.right is None:
                        current.right = new_node
                        #print("This is the right:",current.right.value)
                        break
                    current = current.right

    def _graphize(self, T: nx.Graph, node: Optional[Tree.TreeNode], x: int = 0, y: int = 0, layer_height: Optional[int] = None, layer_width: Optional[int] = None) -> None:
        if node is None:
            return

        if layer_height is None:
            layer_height = random.randint(3, 6)  # Random height for each layer
        if layer_width is None:
            layer_width = 2

        current_pos = (x, y)
        self.pos[node.value] = current_pos

        T.add_node(node.value, value=node.value)

        if node.left:
            T.add_edge(node.value, node.left.value)
            self._graphize(T, node.left, x - layer_width, y - 1, layer_height / 2, layer_width / 2)
        if node.right:
            T.add_edge(node.value, node.right.value)
            self._graphize(T, node.right, x + layer_width, y - 1, layer_height / 2, layer_width / 2)

    def fill(self) -> None:
        """
        Fills the graph nodes with the given values, ensuring the graph reflects the new node values.
        Attempts to refill the tree if the BST properties cannot be maintained within the value bounds.
        """
        while True:
            try:
                used_values = set()
                self._fill_node(self.root, 1, 99, used_values)

                # Clear and recreate the graph to reflect updated node values
                self.graph.clear()
                self.pos.clear()
                self._graphize(self.graph, self.root)
                break  # Break the loop if _fill_node completes without exceptions
            except ValueError:
                # If an exception is caught, the tree filling starts over
                continue

    def _fill_node(self, node: Optional[Tree.TreeNode], min_val: int, max_val: int, used_values: set) -> None:
        if node is None:
            return

        possible_values = set(range(min_val, max_val + 1)) - used_values
        if not possible_values:
            raise ValueError("No available values within the specified range")

        new_value = random.choice(list(possible_values))
        node.value = new_value
        used_values.add(new_value)

        # Recursively update left and right subtrees with updated bounds
        # and prepare to catch exceptions if bounds are violated
        self._fill_node(node.left, min_val, new_value - 1, used_values)
        self._fill_node(node.right, new_value + 1, max_val, used_values)

    def draw(self, save: bool = False, path: Optional[Any] = None, show: bool = True, shape: Shape = 'o', color: Color = '#abe0f9', font: Font = 'sans-serif', width: Width = '1.0', resolution: int = 512, arrow_style: str = '-') -> None:
        """
        Draw the binary search tree
        
        Parameters
        ----------
        save : bool
            whether to save the image or not (default is False)
        path : Optional[Any]
            the path to save the image (default is None)
        show : bool
            whether to show the image or not (default is True)
        shape : Shape
            the shape of the nodes (default is 'o')
        color : Color
            the color of the nodes (default is '#abe0f9' aka sky blue)
        font : Font
            the font of the labels (default is 'sans-serif')
        width : Width
            the width of the edges (default is '1.0')
        
        Raises
        ------
        ValueError
            if the root is None
            
        Notes
        -----
        The visualization is done using the networkx and matplotlib libraries.
        
        The nodes are labeled with their values.
        """
        if self.root is None:
            raise ValueError("The root is None")

        # DPI for the output
        dpi = 100

        # Calculate the figure size in inches for a 512x512 pixel image
        figure_size = resolution / dpi  # 5.12 when dpi is 100
        scale_factor = figure_size / (512 / dpi)

        # Adjusted sizes
        node_size = 800 * (scale_factor ** 2)
        font_size = 12 * scale_factor
        edge_width = float(width) * scale_factor
        
        # Create a figure with the calculated size
        plt.figure(figsize=(figure_size, figure_size))
        
        nx.draw(self.graph, self.pos, with_labels=True, node_size=node_size, node_color=color, font_weight='bold', node_shape=shape, font_family=font, font_size=font_size, linewidths=edge_width, width=edge_width, alpha=1.0, edgecolors=hex_to_rgb_float(color, -50))
        
        if save:
            plt.savefig(fname=path if path else self.default_file_name, format='png', dpi=dpi)
        if show:
            plt.show()
            
        plt.close()
