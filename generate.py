from generators.tree import BinaryTree, BinarySearchTree, generate_binary_tree, generate_binary_search_tree
from generators.graph import UndirectedGraph, DirectedGraph, generate_undirected_graph, generate_directed_graph

binary_tree_path = 'images/binary_tree/test.png'
binary_search_tree_path = 'images/binary_search_tree/'
undirected_graph_path = 'images/undirected_graph/'
directed_graph_path = 'images/directed_graph/'

generation_number = 1
variation_number = 1
format_number = 1

generate_binary_tree(
    large=False,
    save=True,
    path=binary_tree_path,
    filename=f'bit_{generation_number}_{variation_number}_{format_number}.png',
    show=False
)

generate_binary_search_tree(
    large=False,
    save=True,
    path=binary_search_tree_path,
    filename=f'bst_{generation_number}_{variation_number}_{format_number}.png', 
    show=False
)