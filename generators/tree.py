import networkx as nx
import matplotlib.pyplot as plt
import random

class TreeNode:
    def __init__(self, value: int):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self, large: bool = False):
        self.large = large

    def generate(self):
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


    def graphize(self, T, node, pos, x=0, y=0, layer_height=None, layer_width=None):
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
                
    def draw(self, root: TreeNode, save: bool = False, path: str = None, show: bool = True):
        
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
    def __init__(self, large: bool = False):
        self.large = large

    def generate(self):
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

    def graphize(self, T, node, pos, x=0, y=0, layer_height=None, layer_width=None):
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

    def draw(self, root: TreeNode, save: bool = False, path: str = None, show: bool = True):
        
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