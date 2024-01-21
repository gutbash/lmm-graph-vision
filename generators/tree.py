import networkx as nx
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, value: int):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self, large: bool = False):
        self.large = large

    def generate(self):
        if self.large:
            num_nodes = random.randint(10, 20)
        else:
            num_nodes = random.randint(1, 10)

        if num_nodes <= 0:
            return None

        root = Node(1)  # The initial node is consistently designated as 1.
        nodes = [root]

        for i in range(2, num_nodes + 1):
            parent = random.choice(nodes)
            new_node = Node(i)

            if not parent.left:
                parent.left = new_node
            elif not parent.right:
                parent.right = new_node

            nodes.append(new_node)

        return root


    def graphize(self, T, node, pos, x=0, y=0, layer_height=None,layer_width=None):
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
                
    def draw(self, root):
        # Create a directed graph
        T = nx.Graph()
        pos = {}
        # Draw the tree
        self.graphize(T, root, pos)

        # Draw nodes and edges
        nx.draw(T, pos, with_labels=True, font_weight='bold', node_size=800, node_color='skyblue')
        plt.show()
        
class BinarySearchTree:
    def __init__(self, large: bool = False):
        self.large = large

    def generate(self):
        if self.large:
            num_nodes = random.randint(10, 20)
        else:
            num_nodes = random.randint(1, 10)

        if num_nodes <= 0:
            return None

        root_value = random.randint(1, 100)  # Select the root randomly
        root = Node(root_value)
        nodes = [root]

        for i in range(2, num_nodes + 1):
            node_value = random.randint(1, 100)
            new_node = Node(node_value)

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

    def graphize(self, T, node, pos, x=0, y=0, layer_height=None,layer_width=None):
        if node:
            if layer_height is None:
                layer_height = random.randint(20, 30)  # Random height for each layer (increased for a deeper tree)
            if layer_width is None:
                layer_width = 2
                
            pos[node.value] = (x, y)
            if node.left:
                T.add_edge(node.value, node.left.value)
                self.graphize(T, node.left, pos, x - layer_height, y - 1, layer_height / 2, layer_width / 2)
            if node.right:
                T.add_edge(node.value, node.right.value)
                self.graphize(T, node.right, pos, x + layer_height, y - 1, layer_height / 2, layer_width / 2)

    def draw(self, root):
        # Create a directed graph
        T = nx.Graph()

        pos = {}
        # Draw the tree
        self.graphize(T, root, pos)

        # Draw nodes and edges
        nx.draw(T, pos, with_labels=True, font_weight='bold', node_size=800, node_color='skyblue')
        plt.show()

# Instantiate a binary tree
binary_tree = BinaryTree()
# Generate a binary tree 
binary_tree_root = binary_tree.generate()
# Draw the binary tree
#binary_tree.draw(binary_tree_root)

# Instantiate a binary search tree
binary_search_tree = BinarySearchTree()
# Generate a binary search tree
binary_search_tree_root = binary_search_tree.generate()
# Draw the binary search tree
#binary_search_tree.draw(binary_search_tree_root)
