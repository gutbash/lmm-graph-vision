"""
The tree module contains classes and methods for generating random binary trees and binary search trees.
"""

import base64
import networkx as nx
import matplotlib.pyplot as plt
import random

class TreeNode:
    """
    A class used to represent a tree node

    Attributes
    ----------
    value : int
        the value of the node
    left : TreeNode
        the left child of the node
    right : TreeNode
        the right child of the node
    """
    def __init__(self, value: int = None):
        """
        Parameters
        ----------
        value : int
            the value of the node (default is None)
        """
        self.value = value # Node value
        self.left = None # Left child
        self.right = None # Right child

# Generate a random binary tree
class BinaryTree:
    """
    A class used to represent a binary tree

    Attributes
    ----------
    large : bool
        whether to generate a large tree or not

    Methods
    -------
    generate()
        Generates a random binary tree
    graphize(T, node, pos, x=0, y=0, layer_height=None, layer_width=None)
        Graphizes the binary tree
    draw(root: TreeNode, save: bool = False, path: str = None, show: bool = True)
        Draws the binary tree
    """
    
    def __init__(self, large: bool = False):
        """
        Parameters
        ----------
        large : bool
            whether to generate a large tree with 11-20 nodes instead of 1-10 nodes or not (default is False)
        """
        self.large = large

    def generate(self):
        """
        Generates a random binary tree
        
        Paramters
        ---------
        None
        
        Returns
        -------
        TreeNode
            the root of the binary tree
        
        Raises
        ------
        None
        """
        if self.large:
            num_nodes = random.randint(11, 20)
        else:
            num_nodes = random.randint(1, 10)

        if num_nodes <= 0:
            return None

        root_value = random.randint(1, 100) # Prick random root
        root = TreeNode(root_value)
        values_set = {root_value}
        nodes = [root]
        queue = [root]

        while len(nodes) < num_nodes:
            current = queue.pop(0)

            left_child_value = random.randint(1, 100)
            while left_child_value in values_set:
                left_child_value = random.randint(1, 100)

            left_child = TreeNode(left_child_value)
            current.left = left_child
            values_set.add(left_child_value)
            nodes.append(left_child)
            queue.append(left_child)

            if len(nodes) < num_nodes:
                right_child_value = random.randint(1, 100)
                while right_child_value in values_set:
                    right_child_value = random.randint(1, 100)

                right_child = TreeNode(right_child_value)
                current.right = right_child
                values_set.add(right_child_value)
                nodes.append(right_child)
                queue.append(right_child)

        return root

    def graphize(self, T: nx.Graph, node: TreeNode, pos: dict, x: int = 0, y: int = 0, layer_height: int = None, layer_width: int = None):
        """
        Graphizes the binary tree

        Parameters
        ----------
        T : nx.Graph
            the graph to be drawn
        node : TreeNode
            the current node
        pos : dict
            a dictionary of positions of nodes
        x : int
            the x coordinate of the current node (default is 0)
        y : int
            the y coordinate of the current node (default is 0)
        layer_height : int
            the height of the current layer (default is None)
        layer_width : int
            the width of the current layer (default is None)

        Returns
        -------
        None
        
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
            pos[node.value] = (x, y)
            if node.left:
                T.add_edge(node.value, node.left.value)
                self.graphize(T, node.left, pos, x - layer_height, y - 1, layer_height / 2, layer_width / 2)
            if node.right:
                T.add_edge(node.value, node.right.value)
                self.graphize(T, node.right, pos, x + layer_height, y - 1, layer_height / 2, layer_width / 2)
        else:
            raise ValueError("The node is None")
                
    def draw(self, root: TreeNode, save: bool = False, path: str = None, show: bool = True):
        """
        Draws the binary tree using matplotlib and networkx

        Parameters
        ----------
        root : TreeNode
            the root of the binary tree
        save : bool
            whether to save the image or not (default is False)
        path : str
            the path to save the image (default is None)
        show : bool
            whether to show the image or not (default is True)
            
        Returns
        -------
        None

        Raises
        ------
        ValueError
            if the root is None
        """
        
        if root is None:
            raise ValueError("The root is None")
        
        # DPI for the output
        dpi = 100
        
        # Calculate the figure size in inches for a 512x512 pixel image
        figure_size = 512 / dpi  # 5.12 when dpi is 100
        
        # Create a directed graph
        T = nx.Graph()
        pos = {}
        
        # Draw the tree
        self.graphize(T, root, pos)
        
        # Create a figure with the calculated size
        plt.figure(figsize=(figure_size, figure_size))

        # Draw nodes and edges
        nx.draw(T, pos, with_labels=True, font_weight='bold', node_size=800, node_color='skyblue')

        if save:
            if path is None:
                path = "output.png"  # Default file name
            plt.savefig(fname=path, format='png', dpi=dpi)
            
        if show:
            plt.show()
        
class BinarySearchTree:
    """
    A class used to represent a binary search tree

    Attributes
    ----------
    large : bool
        whether to generate a large tree or not

    Methods
    -------
    generate()
        Generates a random binary search tree
    graphize(T, node, pos, x=0, y=0, layer_height=None, layer_width=None)
        Graphizes the binary search tree
    draw(root: TreeNode, save: bool = False, path: str = None, show: bool = True)
        Draws the binary search tree
    """    

    def __init__(self, large: bool = False):
        """
        Parameters
        ----------
        large : bool
            whether to generate a large tree with 11-20 nodes instead of 1-10 nodes or not (default is False)
        """
        self.large = large

    def generate(self):
        """
        Generates a random binary search tree
        
        Parameters
        ----------
        None
        
        Returns
        -------
        TreeNode
            the root of the binary search tree
            
        Raises
        ------
        None
        """
        if self.large:
            num_nodes = random.randint(11, 20)
        else:
            num_nodes = random.randint(1, 10)

        if num_nodes <= 0:
            return None

        root_value = random.randint(1, 100)  # Select the root randomly
        root = TreeNode(root_value)
        nodes = [root]

        for i in range(2, num_nodes + 1):
            node_value = random.randint(1, 100)
            new_node = TreeNode(node_value)

            current = root
            while True:
                if new_node.value < current.value:
                    if current.left is None:
                        current.left = new_node
                        break
                    else:
                        current = current.left
                elif new_node.value > current.value:
                    if current.right is None:
                        current.right = new_node
                        break
                    else:
                        current = current.right

            nodes.append(new_node)

        return root

    def graphize(self, T: nx.Graph, node: TreeNode, pos: dict, x: int = 0, y: int = 0, layer_height: int = None, layer_width: int = None):
        """
        Graphizes the binary search tree

        Parameters
        ----------
        T : nx.Graph
            the graph to be drawn
        node : TreeNode
            the current node
        pos : dict
            a dictionary of positions of nodes
        x : int
            the x coordinate of the current node (default is 0)
        y : int
            the y coordinate of the current node (default is 0)
        layer_height : int
            the height of the current layer (default is None)
        layer_width : int
            the width of the current layer (default is None)

        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            if the node is None
        """
        if node:
            if layer_height is None:
                layer_height = random.uniform(0.5, 1.5)  # Random height for each layer (increased for a deeper tree)
            if layer_width is None:
                layer_width = 1.0
                
            pos[node.value] = (x, y)
            if node.left:
                T.add_edge(node.value, node.left.value)
                self.graphize(T, node.left, pos, x - layer_height, y - 1, layer_height / 2, layer_width / 2)
            if node.right:
                T.add_edge(node.value, node.right.value)
                self.graphize(T, node.right, pos, x + layer_height, y - 1, layer_height / 2, layer_width / 2)
        else:
            raise ValueError("The node is None")

    def draw(self, root: TreeNode, save: bool = False, path: str = None, show: bool = True):
        """
        Draws the binary search tree using matplotlib and networkx

        Parameters
        ----------
        root : TreeNode
            the root of the binary search tree
        save : bool
            whether to save the image or not (default is False)
        path : str
            the path to save the image (default is None)
        show : bool
            whether to show the image or not (default is True)
            
        Returns
        -------
        None

        Raises
        ------
        ValueError
            if the root is None
        """
        
        if root is None:
            raise ValueError("The root is None")
        
        # DPI for the output
        dpi = 100
        
        # Calculate the figure size in inches for a 512x512 pixel image
        figure_size = 512 / dpi  # 5.12 when dpi is 100
        
        # Create a directed graph
        T = nx.Graph()
        pos = nx.spring_layout(T)
        
        # Draw the tree
        self.graphize(T, root, pos)
        
        # Create a figure with the calculated size
        plt.figure(figsize=(figure_size, figure_size))

        # Draw nodes and edges
        nx.draw(T, pos, with_labels=True, font_weight='bold', node_size=800, node_color='skyblue')
        
        if save:
            if path is None:
                path = "output.png"  # Default file name
            plt.savefig(fname=path, format='png', dpi=dpi)
            
        if show:
            plt.show()