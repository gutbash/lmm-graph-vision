from generation.generator import Generator, BatchGenerator
from generation.structures.tree import BinaryTree, BinarySearchTree
from generation.structures.graph import UndirectedGraph, DirectedGraph

from evaluation.evaluator import Evaluator
from evaluation.models.openai import OpenAI
from evaluation.models.deepmind import DeepMind

from evaluation.models.messages.message import UserMessage, SystemMessage, AssistantMessage, ImageMessage, BaseMessage

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from uuid import uuid4
import cProfile
import pstats

load_dotenv()

openai_api_key = os.environ.get('OPENAI_API_KEY_DEV')
deepmind_api_key = os.environ.get('DEEPMIND_API_KEY_DEV')

### DEVELOP PATHS ###

image_path_binary_tree = Path('images/binary_tree/')
image_path_binary_search_tree = Path('images/binary_search_tree/')
image_path_undirected_graph = Path('images/undirected_graph/')
image_path_directed_graph = Path('images/directed_graph/')

yaml_path = Path('data/')
text_path = Path('text/')

### TEST BATCH GENERATION ###

batch_generator = BatchGenerator()
generation = 5
variation = 3

async def run_batch():
    '''
    await batch_generator.generate_batch(
        structure_class=BinaryTree,
        type='bit',
        yaml_name='binary_tree.yaml',
        yaml_path=yaml_path,
        save_path=image_path_binary_tree,
        text_path=text_path,
        text_name='binary_tree_text.yaml',
        generations=generation,
        variations=variation,
        random_num_nodes=False,
        resolutions=[256, 512, 1024, 2048],
        visual_combinations=False,
    )
    '''
    await batch_generator.generate_batch(
        structure_class=BinarySearchTree,
        type='bst',
        yaml_name='binary_search_tree.yaml',
        yaml_path=yaml_path,
        save_path=image_path_binary_search_tree,
        text_path=text_path,
        text_name='binary_search_tree_text.yaml',
        generations=generation,
        variations=variation,
        random_num_nodes=False,
        resolutions=[256, 512, 1024, 2048],
        visual_combinations=False,
    )
    '''
    await batch_generator.generate_batch(
        structure_class=UndirectedGraph,
        type='ug',
        yaml_name='undirected_graph.yaml',
        yaml_path=yaml_path,
        save_path=image_path_undirected_graph,
        text_path=text_path,
        text_name='undirected_graph_text.yaml',
        generations=generation,
        variations=variation,
        random_num_nodes=False,
        resolutions=[256, 512, 1024, 2048],
        visual_combinations=False,
    )

    await batch_generator.generate_batch(
        structure_class=DirectedGraph,
        type='dg',
        yaml_name='directed_graph.yaml',
        yaml_path=yaml_path,
        save_path=image_path_directed_graph,
        text_path=text_path,
        text_name='directed_graph_text.yaml',
        generations=generation,
        variations=variation,
        random_num_nodes=False,
        resolutions=[256, 512, 1024, 2048],
        visual_combinations=False,
    )
    '''
asyncio.run(run_batch())
    
### TEST EVALUATION ###

openai = OpenAI(api_key=openai_api_key)
deepmind = DeepMind(api_key=deepmind_api_key)

openai_messages = [
    UserMessage(content="{{content}}", images=["{{image}}"]),
]

deepmind_messages = [
    BaseMessage(content="{{content}}"),
    ImageMessage(image="{{image}}"),
]

openai_csv = 'openai.csv'
deepmind_csv = f'deepmind-{uuid4()}.csv'

evaluator = Evaluator()

model = deepmind
csv_name = deepmind_csv
messages = deepmind_messages

async def run_eval():

    for structure in ['binary_tree', 'undirected_graph', 'directed_graph']:

        await evaluator.evaluate(model=model, messages=messages, yaml_path=yaml_path, yaml_name=f'{structure}.yaml', csv_path=Path('results/'), csv_name=csv_name, repeats=3)

#asyncio.run(run_eval())