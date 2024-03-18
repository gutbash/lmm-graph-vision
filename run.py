from generation.generator import Generator, BatchGenerator
from generation.structures.tree import BinaryTree, BinarySearchTree
from generation.structures.graph import UndirectedGraph, DirectedGraph

from evaluation.evaluator import Evaluator
from evaluation.models.openai import OpenAI
from evaluation.models.deepmind import DeepMind

from utils.logger import Logger
logger = Logger(__name__)

from evaluation.prompts import ZERO_SHOT_OLD, REPHRASE, NO_STRUCTURE, STEPS, DEFINITION, EXPERT, ZERO_SHOT_COT_REREAD, SERIAL, POLITE, ZERO_SHOT_COT, GOLD_COT, GENERAL_KNOWLEDGE, ROLEPLAY_EXPERT_COT, DELIMIT, GOLD_COT_EXPERT_DELIMIT, ZERO_SHOT_COT_POLITE, ZERO_SHOT_ROOT_ATTENTION, ZERO_SHOT

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from uuid import uuid4
import cProfile
import pstats

load_dotenv()

### keys ###

openai_api_key = os.environ.get('OPENAI_API_KEY_DEV')
deepmind_api_key = os.environ.get('DEEPMIND_API_KEY_DEV')

### prompts ###

PROMPTS = {'zero_shot': ZERO_SHOT}

### paths ###

image_path_binary_tree = Path('images/binary_tree/')
image_path_binary_search_tree = Path('images/binary_search_tree/')
image_path_undirected_graph = Path('images/undirected_graph/')
image_path_directed_graph = Path('images/directed_graph/')

yaml_path = Path('data/')

### combinations ###

COLORS = ['#ffffff', '#ffff00', '#ff0000'] # white, yellow, red, green, blue
SHAPES = ['o', 's', 'd']
FONTS = ['sans-serif', 'serif', 'monospace']
WIDTH = ['1.0', '3.0', '5.0']
ARROWS = ['->', '-|>']
RESOLUTIONS = [256, 512, 1024, 2048]
STRUCTURES = ['binary_tree', 'binary_search_tree', 'undirected_graph', 'directed_graph']

###### test generation ######

batch_generator = BatchGenerator()
generation = 3
variation = 3

async def run_batch():
    
    await batch_generator.generate_batch(
        structure_class=BinaryTree,
        type='bit',
        yaml_name='binary_tree.yaml',
        yaml_path=yaml_path,
        save_path=image_path_binary_tree,
        generations=generation,
        variations=variation,
        colors=COLORS,
        width=WIDTH,
    )
    
    await batch_generator.generate_batch(
        structure_class=BinarySearchTree,
        type='bst',
        yaml_name='binary_search_tree.yaml',
        yaml_path=yaml_path,
        save_path=image_path_binary_search_tree,
        generations=generation,
        variations=variation,
        colors=COLORS,
        width=WIDTH,
    )
    
    await batch_generator.generate_batch(
        structure_class=UndirectedGraph,
        type='ug',
        yaml_name='undirected_graph.yaml',
        yaml_path=yaml_path,
        save_path=image_path_undirected_graph,
        generations=generation,
        variations=variation,
        colors=COLORS,
        width=WIDTH,
    )
    
    await batch_generator.generate_batch(
        structure_class=DirectedGraph,
        type='dg',
        yaml_name='directed_graph.yaml',
        yaml_path=yaml_path,
        save_path=image_path_directed_graph,
        generations=generation,
        variations=variation,
        colors=COLORS,
        width=WIDTH,
    )

###### test evaluation ######

openai = OpenAI(api_key=openai_api_key)
deepmind = DeepMind(api_key=deepmind_api_key)

openai_csv = f'openai'
deepmind_csv = f'deepmind'

evaluator = Evaluator()

model = deepmind
csv_name = deepmind_csv

eval_name = 'large_course'

async def run_eval():
    
    for prompt_name, prompts in PROMPTS.items():

        for structure in ['binary_tree', 'binary_search_tree', 'undirected_graph', 'directed_graph']:
            
            try:

                await evaluator.evaluate(model=model, prompts=prompts, yaml_path=yaml_path, yaml_name=f'{structure}.yaml', csv_path=Path('results/'), csv_name=f'{csv_name}-{prompt_name}-{eval_name}.csv', repeats=3)
                
            except Exception as e:
                logger.error(f'{e}')
                return
            
#asyncio.run(run_batch())
asyncio.run(run_eval())

#TODO: control graph edge count
#TODO: control node values between structures

# how many edges?
# how many vertices?
# degree of node n?
# shortest path?

#TODO: showcase visual parameter influences
#TODO: showcase node value influences
#TODO: showcase arrow style influences
#TODO: showcase edge number influences