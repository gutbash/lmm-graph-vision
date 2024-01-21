import networkx as nx
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def generate_binary_search_tree(num_nodes):
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

def tree_to_graph(T, node, pos, x=0, y=0, layer_height=None,layer_width=None):
    if node:
        if layer_height is None:
            layer_height = random.randint(20, 30)  # Random height for each layer (increased for a deeper tree)
        if layer_width is None:
            layer_width = 2
            
        pos[node.value] = (x, y)
        if node.left:
            T.add_edge(node.value, node.left.value)
            tree_to_graph(T, node.left, pos, x - layer_height, y - 1, layer_height / 2, layer_width / 2)
        if node.right:
            T.add_edge(node.value, node.right.value)
            tree_to_graph(T, node.right, pos, x + layer_height, y - 1, layer_height / 2, layer_width / 2)

# Specify the number of nodes in the binary search tree
num_nodes = random.randint(10, 20)

# Generate a binary search tree 
root = generate_binary_search_tree(num_nodes)

# Create a directed graph
T = nx.DiGraph()

pos = {}
# Draw the tree
tree_to_graph(T, root, pos)

# Draw nodes and edges
nx.draw(T, pos, with_labels=True, font_weight='bold', connectionstyle='arc3,rad=0', node_size=800, node_color='skyblue')
plt.savefig("BST.png", format= "png")
plt.show()
