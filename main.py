from generators.generation import Generator, BatchGenerator
from generators.structures.tree import BinaryTree, BinarySearchTree
from generators.structures.graph import UndirectedGraph, DirectedGraph

from evaluation import Evaluator

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get('OPENAI_API_KEY')

### DEVELOP PATHS ###

image_path_binary_tree = Path('images/develop/binary_tree/')
image_path_binary_search_tree = Path('images/develop/binary_search_tree/')
image_path_undirected_graph = Path('images/develop/undirected_graph/')
image_path_directed_graph = Path('images/develop/directed_graph/')

yaml_path_binary_tree = Path('data/develop/binary_tree/')
yaml_path_binary_search_tree = Path('data/develop/binary_search_tree/')
yaml_path_undirected_graph = Path('data/develop/undirected_graph/')
yaml_path_directed_graph = Path('data/develop/directed_graph/')

text_path_binary_tree = Path('text/develop/binary_tree/')
text_path_binary_search_tree = Path('text/develop/binary_search_tree/')
text_path_undirected_graph = Path('text/develop/undirected_graph/')
text_path_directed_graph = Path('text/develop/directed_graph/')

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

### TEST GENERATION ###
'''
generator = Generator()

structure = generator.generate_structure(
    structure_class=BinaryTree,
    large=False,
)

filled = generator.fill_structure(
    structure_instance=structure,
)

inorder = filled.traversal('inorder')
postorder = filled.traversal('postorder')
preorder = filled.traversal('preorder')

print(inorder, postorder, preorder)

generator.draw_structure(
    structure_instance=structure,
    yaml=False,
    yaml_path=yaml_path_binary_tree,
    yaml_name='binary_tree.yaml',
    save=True,
    save_path=image_path_binary_tree,
    save_name='binary_tree.png',
    show=False,
    run=0,
    generation=0,
    variation=0,
    format=0,
    shape='o',
    color='#88d7fe',
    font='sans-serif',
)

structure = generator.generate_structure(
    structure_class=BinarySearchTree,
    large=False,
)

filled = generator.fill_structure(
    structure_instance=structure,
)

generator.draw_structure(
    structure_instance=structure,
    yaml=False,
    yaml_path=yaml_path_binary_search_tree,
    yaml_name='binary_search_tree.yaml',
    save=True,
    save_path=image_path_binary_search_tree,
    save_name='binary_search_tree.png',
    show=False,
    run=0,
    generation=0,
    variation=0,
    format=0,
    shape='o',
    color='#88d7fe',
    font='sans-serif',
)

structure = generator.generate_structure(
    structure_class=UndirectedGraph,
    large=False,
)

filled = generator.fill_structure(
    structure_instance=structure,
)

generator.draw_structure(
    structure_instance=structure,
    yaml=False,
    yaml_path=yaml_path_undirected_graph,
    yaml_name='undirected_graph.yaml',
    save=True,
    save_path=image_path_undirected_graph,
    save_name='undirected_graph.png',
    show=False,
    run=0,
    generation=0,
    variation=0,
    format=0,
    shape='o',
    color='#88d7fe',
    font='sans-serif',
)

structure = generator.generate_structure(
    structure_class=DirectedGraph,
    large=False,
)

filled = generator.fill_structure(
    structure_instance=structure,
)

generator.draw_structure(
    structure_instance=structure,
    yaml=False,
    yaml_path=yaml_path_directed_graph,
    yaml_name='directed_graph.yaml',
    save=True,
    save_path=image_path_directed_graph,
    save_name='directed_graph.png',
    show=False,
    run=0,
    generation=0,
    variation=0,
    format=0,
    shape='o',
    color='#88d7fe',
    font='sans-serif',
)
'''
### TEST BATCH GENERATION ###


batch_generator = BatchGenerator()

batch_generator.generate_batch(
    structure_class=BinaryTree,
    type='bit',
    yaml_name='binary_tree.yaml',
    yaml_path=yaml_path_binary_tree,
    save_path=image_path_binary_tree,
    text_path=text_path_binary_tree,
    text_name='binary_tree_text.yaml',
)

batch_generator.generate_batch(
    structure_class=BinarySearchTree,
    type='bst',
    yaml_name='binary_search_tree.yaml',
    yaml_path=yaml_path_binary_search_tree,
    save_path=image_path_binary_search_tree,
    text_path=text_path_binary_search_tree,
    text_name='binary_search_tree_text.yaml',
)

batch_generator.generate_batch(
    structure_class=UndirectedGraph,
    type='ug',
    yaml_name='undirected_graph.yaml',
    yaml_path=yaml_path_undirected_graph,
    save_path=image_path_undirected_graph,
    text_path=text_path_undirected_graph,
    text_name='undirected_graph_text.yaml',
)

batch_generator.generate_batch(
    structure_class=DirectedGraph,
    type='dg',
    yaml_name='directed_graph.yaml',
    yaml_path=yaml_path_directed_graph,
    save_path=image_path_directed_graph,
    text_path=text_path_directed_graph,
    text_name='directed_graph_text.yaml',
)

### TEST EVALUATION ###
'''
evaluator = Evaluator(
    api_key=api_key,
)

evaluator.evaluate(
    limit=1,
    path=yaml_path_binary_tree,
    filename='binary_tree.yaml',
)

evaluator.evaluate(
    limit=1,
    path=yaml_path_binary_search_tree,
    filename='binary_search_tree.yaml',
)

evaluator.evaluate(
    limit=1,
    path=yaml_path_undirected_graph,
    filename='undirected_graph.yaml',
)

evaluator.evaluate(
    limit=1,
    path=yaml_path_directed_graph,
    filename='directed_graph.yaml',
)
'''