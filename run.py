from generation.generator import Generator, BatchGenerator
from generation.structures.tree import BinaryTree, BinarySearchTree
from generation.structures.graph import UndirectedGraph, DirectedGraph

from evaluation.evaluator import Evaluator
from evaluation.models.openai import OpenAI
from evaluation.models.deepmind import DeepMind

from utils.logger import Logger
logger = Logger(__name__)

from evaluation.prompts import PROMPTS_ZERO_SHOT, PROMPTS_REPHRASE, PROMPTS_NO_STRUCTURE, PROMPTS_STEPS, PROMPTS_DEFINITION, PROMPTS_EXPERT, PROMPTS_ZERO_SHOT_COT_REREAD, PROMPTS_SERIAL, PROMPTS_POLITE, PROMPTS_ZERO_SHOT_COT, PROMPTS_GOLD_COT, PROMPTS_GENERAL_KNOWLEDGE, PROMPTS_ROLEPLAY_EXPERT_COT, PROMPTS_DELIMIT, PROMPTS_GOLD_COT_EXPERT_DELIMIT, PROMPTS_ZERO_SHOT_COT_POLITE

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

PROMPTS = {'prompts_serial': PROMPTS_SERIAL, 'prompts_zero_shot': PROMPTS_ZERO_SHOT, 'prompts_rephrase': PROMPTS_REPHRASE, 'prompts_no_structure': PROMPTS_NO_STRUCTURE, 'prompts_steps': PROMPTS_STEPS, 'prompts_definition': PROMPTS_DEFINITION, 'prompts_expert': PROMPTS_EXPERT, 'prompts_zero_shot_cot_reread': PROMPTS_ZERO_SHOT_COT_REREAD, 'prompts_polite': PROMPTS_POLITE, 'prompts_zero_shot_cot': PROMPTS_ZERO_SHOT_COT, 'prompts_gold_cot': PROMPTS_GOLD_COT, 'prompts_general_knowledge': PROMPTS_GENERAL_KNOWLEDGE, 'prompts_roleplay_expert_cot': PROMPTS_ROLEPLAY_EXPERT_COT, 'prompts_delimit': PROMPTS_DELIMIT, 'prompts_gold_cot_expert_delimit': PROMPTS_GOLD_COT_EXPERT_DELIMIT}

PROMPTS = {'prompts_zero_shot': PROMPTS_ZERO_SHOT, 'prompts_zero_shot_cot': PROMPTS_ZERO_SHOT_COT}

### DEVELOP PATHS ###

image_path_binary_tree = Path('images/binary_tree/')
image_path_binary_search_tree = Path('images/binary_search_tree/')
image_path_undirected_graph = Path('images/undirected_graph/')
image_path_directed_graph = Path('images/directed_graph/')

yaml_path = Path('data/')

### TEST BATCH GENERATION ###

batch_generator = BatchGenerator()
generation = 7
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

openai_csv = f'openai'
deepmind_csv = f'deepmind'

evaluator = Evaluator()

model = deepmind
csv_name = deepmind_csv

async def run_eval():
    
    for prompt_name, prompts in PROMPTS.items():

        for structure in ['binary_tree', 'binary_search_tree', 'undirected_graph', 'directed_graph']:
            
            try:

                await evaluator.evaluate(model=model, prompts=prompts, yaml_path=yaml_path, yaml_name=f'{structure}.yaml', csv_path=Path('results/'), csv_name=f'{csv_name}-{prompt_name}-duel.csv', repeats=3)
                
            except Exception as e:
                logger.error(f'{e}')
                return

asyncio.run(run_eval())