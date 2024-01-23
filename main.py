from generators.generation import generate_structure
from generators.structures.tree import BinaryTree, BinarySearchTree
from generators.structures.graph import UndirectedGraph, DirectedGraph

from pathlib import Path

image_path_binary_tree = Path('images/binary_tree/')
image_path_binary_search_tree = Path('images/binary_search_tree/')
image_path_undirected_graph = Path('images/undirected_graph/')
image_path_directed_graph = Path('images/directed_graph/')

yaml_path_binary_tree = Path('data/binary_tree/')
yaml_path_binary_search_tree = Path('data/binary_search_tree/')
yaml_path_undirected_graph = Path('data/undirected_graph/')
yaml_path_directed_graph = Path('data/directed_graph/')

generation_number = 0
variation_number = 0
format_number = 0

### BEGIN TESTING AREA ###

generate_structure(
    structure=BinaryTree,
    large=False,
    yaml=False,
    yaml_path=yaml_path_binary_tree,
    yaml_name='binary_tree.yaml',
    save=True,
    save_path=image_path_binary_tree,
    file_name='binary_tree.png',
    show=False,
    generation=generation_number,
    variation=variation_number,
    format=format_number
)

generate_structure(
    structure=BinarySearchTree,
    large=False,
    yaml=False,
    yaml_path=yaml_path_binary_search_tree,
    yaml_name='binary_search_tree.yaml',
    save=True,
    save_path=image_path_binary_search_tree,
    file_name='binary_search_tree.png',
    show=False,
    generation=generation_number,
    variation=variation_number,
    format=format_number
)

generate_structure(
    structure=UndirectedGraph,
    large=False,
    yaml=False,
    yaml_path=yaml_path_undirected_graph,
    yaml_name='undirected_graph.yaml',
    save=True,
    save_path=image_path_undirected_graph,
    file_name='undirected_graph.png',
    show=False,
    generation=generation_number,
    variation=variation_number,
    format=format_number
)

generate_structure(
    structure=DirectedGraph,
    large=False,
    yaml=False,
    yaml_path=yaml_path_directed_graph,
    yaml_name='directed_graph.yaml',
    save=True,
    save_path=image_path_directed_graph,
    file_name='directed_graph.png',
    show=False,
    generation=generation_number,
    variation=variation_number,
    format=format_number
)

### END TESTING AREA ###