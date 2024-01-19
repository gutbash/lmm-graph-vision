import networkx as nx
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def generate_binary_tree(num_nodes):
    if num_nodes <= 0:
        return None
        
    root = Node(1) #The initial node is consistently designated as 1.
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

def tree_to_graph(T, node, pos, x=0, y=0, layer_height=None):
    if node:
        if layer_height is None:
            layer_height = random.randint(3, 6)  # Random height for each layer
        pos[node.value] = (x, y)
        if node.left:
            T.add_edge(node.value, node.left.value)
            tree_to_graph(T, node.left, pos, x - layer_height, y - 1, layer_height / 2)
        if node.right:
            T.add_edge(node.value, node.right.value)
            tree_to_graph(T, node.right, pos, x + layer_height, y - 1, layer_height / 2)

# Randomly pick the number of the nodes
num_nodes = random.randint(1,10)

# Generate a binary tree 
root = generate_binary_tree(num_nodes)

# Create a directed graph
T = nx.DiGraph()

pos = {}
# Draw the tree
tree_to_graph(T, root, pos)

# Draw nodes and edges
nx.draw(T, pos, with_labels=True, font_weight='bold', connectionstyle='arc3,rad=0', node_size=800, node_color='skyblue')
plt.savefig("BinaryTree.png", format= "png")
plt.show()
