from generators.structure import generate_structure, fill_structure, draw_structure
from generators.structures.tree import BinaryTree, BinarySearchTree
from generators.structures.graph import UndirectedGraph, DirectedGraph

from pathlib import Path

# TODO: Create iterative dataset generation methods for each structure
# TODO: Separate drawing and saving

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

colors = ['#88d7fe', '#feaf88', '#eeeeee']
shapes = ['o', 's', 'd']
fonts = ['sans-serif', 'serif', 'monospace']

undirected_graph_structure_generated = generate_structure(
    structure_class=UndirectedGraph,
    large=False,
)

undirected_graph_structure_filled = None

for variation in range(1, 4):
    
    undirected_graph_structure_filled = fill_structure(
        structure_instance=undirected_graph_structure_generated,
    )
    
    draw_structure(
        structure_instance=undirected_graph_structure_filled,
        yaml=False,
        save=True,
        yaml_path=yaml_path_undirected_graph,
        yaml_name='undirected_graph.yaml',
        save_path=image_path_undirected_graph,
        file_name=f'undirected_graph_structure_{variation}.png',
        show=False,
        generation=generation_number,
        variation=variation_number,
        format=format_number,
        shape='o',
        color='#88d7fe',
        font='sans-serif',
    )

for color in colors:
    
    draw_structure(
        structure_instance=undirected_graph_structure_filled,
        yaml=False,
        save=True,
        yaml_path=yaml_path_undirected_graph,
        yaml_name='undirected_graph.yaml',
        save_path=image_path_undirected_graph,
        file_name=f'undirected_graph_structure_{color}.png',
        show=False,
        generation=generation_number,
        variation=variation_number,
        format=format_number,
        shape='o',
        color=color,
        font='sans-serif',
    )
    
for shape in shapes:
    
    draw_structure(
        structure_instance=undirected_graph_structure_filled,
        yaml=False,
        save=True,
        yaml_path=yaml_path_undirected_graph,
        yaml_name='undirected_graph.yaml',
        save_path=image_path_undirected_graph,
        file_name=f'undirected_graph_structure_{shape}.png',
        show=False,
        generation=generation_number,
        variation=variation_number,
        format=format_number,
        shape=shape,
        color='#88d7fe',
        font='sans-serif',
    )
    
for font in fonts:
    
    draw_structure(
        structure_instance=undirected_graph_structure_filled,
        yaml=False,
        save=True,
        yaml_path=yaml_path_undirected_graph,
        yaml_name='undirected_graph.yaml',
        save_path=image_path_undirected_graph,
        file_name=f'undirected_graph_structure_{font}.png',
        show=False,
        generation=generation_number,
        variation=variation_number,
        format=format_number,
        shape='o',
        color='#88d7fe',
        font=font,
    )