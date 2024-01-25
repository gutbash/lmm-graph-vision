from generators.generation import generate_structure, fill_structure
from generators.structures.tree import BinaryTree, BinarySearchTree
from generators.structures.graph import UndirectedGraph, DirectedGraph

from pathlib import Path

### DEVELOPMENT PATHS ###

image_path_binary_tree = Path('images/develop/binary_tree/')
image_path_binary_search_tree = Path('images/develop/binary_search_tree/')
image_path_undirected_graph = Path('images/develop/undirected_graph/')
image_path_directed_graph = Path('images/develop/directed_graph/')

yaml_path_binary_tree = Path('data/develop/binary_tree/')
yaml_path_binary_search_tree = Path('data/develop/binary_search_tree/')
yaml_path_undirected_graph = Path('data/develop/undirected_graph/')
yaml_path_directed_graph = Path('data/develop/directed_graph/')

### DEPLOY PATHS ###

"""
image_path_binary_tree = Path('images/deploy/binary_tree/')
image_path_binary_search_tree = Path('images/deploy/binary_search_tree/')
image_path_undirected_graph = Path('images/deploy/undirected_graph/')
image_path_directed_graph = Path('images/deploy/directed_graph/')

yaml_path_binary_tree = Path('data/deploy/binary_tree/')
yaml_path_binary_search_tree = Path('data/deploy/binary_search_tree/')
yaml_path_undirected_graph = Path('data/deploy/undirected_graph/')
yaml_path_directed_graph = Path('data/deploy/directed_graph/')
"""

generation_number = 0
variation_number = 0
format_number = 0

### TEST STRUCTURE GENERATION ###

binary_tree_structure = generate_structure(
    structure=BinaryTree,
    large=False,
)

for variation in range(1, 4):
    
    fill_structure(
        structure_instance=binary_tree_structure,
        yaml=False,
        yaml_path=yaml_path_binary_tree,
        yaml_name='binary_tree.yaml',
        save=True,
        save_path=image_path_binary_tree,
        file_name=f'binary_tree_test_{variation}.png',
        show=False,
        generation=generation_number,
        variation=variation_number,
        format=format_number
    )
    
undirected_graph_structure = generate_structure(
    structure=UndirectedGraph,
    large=False,
)

for variation in range(1, 4):
    
    fill_structure(
        structure_instance=undirected_graph_structure,
        yaml=False,
        yaml_path=yaml_path_undirected_graph,
        yaml_name='undirected_graph.yaml',
        save=True,
        save_path=image_path_undirected_graph,
        file_name=f'undirected_graph_test_{variation}.png',
        show=False,
        generation=generation_number,
        variation=variation_number,
        format=format_number
    )

directed_graph_structure = generate_structure(
    structure=DirectedGraph,
    large=False,
)

for variation in range(1, 4):
        
    fill_structure(
        structure_instance=directed_graph_structure,
        yaml=False,
        yaml_path=yaml_path_directed_graph,
        yaml_name='directed_graph.yaml',
        save=True,
        save_path=image_path_directed_graph,
        file_name=f'directed_graph_test_{variation}.png',
        show=False,
        generation=generation_number,
        variation=variation_number,
        format=format_number
    )

'''
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
'''