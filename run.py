from generation.generator import Generator, BatchGenerator
from generation.structures.tree import BinaryTree, BinarySearchTree
from generation.structures.graph import UndirectedGraph, DirectedGraph

from evaluation.evaluator import Evaluator
from evaluation.models.openai import OpenAI
from evaluation.models.deepmind import DeepMind

from evaluation.prompts import PROMPTS_DEFAULT

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

### TEST BATCH GENERATION ###

batch_generator = BatchGenerator()
generation = 1
variation = 1

async def run_batch():

    await batch_generator.generate_batch(
        structure_class=BinaryTree,
        type='bit',
        yaml_name='binary_tree.yaml',
        yaml_path=yaml_path,
        save_path=image_path_binary_tree,
        generations=generation,
        variations=variation,
        random_num_nodes=False,
        resolutions=[512],
        visual_combinations=False,
    )

    await batch_generator.generate_batch(
        structure_class=BinarySearchTree,
        type='bst',
        yaml_name='binary_search_tree.yaml',
        yaml_path=yaml_path,
        save_path=image_path_binary_search_tree,
        generations=generation,
        variations=variation,
        random_num_nodes=False,
        resolutions=[512],
        visual_combinations=False,
    )

    await batch_generator.generate_batch(
        structure_class=UndirectedGraph,
        type='ug',
        yaml_name='undirected_graph.yaml',
        yaml_path=yaml_path,
        save_path=image_path_undirected_graph,
        generations=generation,
        variations=variation,
        random_num_nodes=False,
        resolutions=[512],
        visual_combinations=False,
    )

    await batch_generator.generate_batch(
        structure_class=DirectedGraph,
        type='dg',
        yaml_name='directed_graph.yaml',
        yaml_path=yaml_path,
        save_path=image_path_directed_graph,
        generations=generation,
        variations=variation,
        random_num_nodes=False,
        resolutions=[512],
        visual_combinations=False,
    )
#asyncio.run(run_batch())
'''
cProfile.run('asyncio.run(run_batch())', 'batch_stats')
p = pstats.Stats('batch_stats')
p.sort_stats('cumulative').print_stats(10)
'''
### TEST EVALUATION ###

openai = OpenAI(api_key=openai_api_key)
deepmind = DeepMind(api_key=deepmind_api_key)

openai_csv = 'openai.csv'
deepmind_csv = f'deepmind-{uuid4()}.csv'

evaluator = Evaluator()

model = openai
csv_name = openai_csv
prompts = PROMPTS_DEFAULT

async def run_eval():

    for structure in ['binary_tree', 'binary_search_tree', 'undirected_graph', 'directed_graph']:

        await evaluator.evaluate(model=model, prompts=prompts, yaml_path=yaml_path, yaml_name=f'{structure}.yaml', csv_path=Path('results/'), csv_name=csv_name, repeats=1)

asyncio.run(run_eval())