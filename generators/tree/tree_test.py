import matplotlib.pyplot as plt
import networkx as nx
import random
from networkx.drawing.nx_agraph import graphviz_layout

def generate_random_binary_tree(n):
    if n <= 0:
        return None

    G = nx.DiGraph()
    G.add_node(0)  # starting with the root node

    for i in range(1, n):
        while True:
            parent = random.choice(list(G.nodes))
            if G.out_degree(parent) < 2:
                G.add_edge(parent, i)
                break

    return G

def draw_binary_tree(G):
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=False, node_size=700, node_color="skyblue", font_size=10)
    plt.show()

# Example usage
n = 10
tree = generate_random_binary_tree(n)
draw_binary_tree(tree)