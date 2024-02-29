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

class Main:
    def __init__(self):
        self.setup = EnvironmentSetup()
        self.batch_gen = BatchGeneration(self.setup)
        self.eval = Evaluation({'openai': self.setup.openai_api_key, 'deepmind': self.setup.deepmind_api_key}, self.setup)

    async def run(self):
        # [GENERATE DATA STRUCTURES] #
        selected_structures = [         # <-- [SELECT: Data Structure Type] 
            BinaryTree,
            # BinarySearchTree,
            # UndirectedGraph,
            # DirectedGraph,
        ]
        generations = 1                 # <-- [SELECT: Generations]
        variations = 1                  # <-- [SELECT: Variations]
        await self.batch_gen.generate_batches(selected_structures, generations, variations)

        # [EVALUATE DATA STRUCTURES] #
        selected_structure_names = [    # <-- [SELECT: Model & Data Structure Type] 
            # DeepMind evaluations #
            ('deepmind', 'binary_tree'),
            #('deepmind', 'binary_search_tree'),
            #('deepmind', 'undirected_graph'),
            #('deepmind', 'directed_graph'),
            
            # OpenAI evaluations #
            #('openai', 'binary_tree'),
            #('openai', 'binary_search_tree'),
            #('openai', 'undirected_graph'),
            #('openai', 'directed_graph'),
        ]
        await self.eval.evaluate_models(selected_structure_names)

class EnvironmentSetup:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY_DEV')
        self.deepmind_api_key = os.getenv('DEEPMIND_API_KEY_DEV')
        self.setup_paths()

    def setup_paths(self):
        self.image_path = {
            'binary_tree': Path('images/binary_tree/'),
            'binary_search_tree': Path('images/binary_search_tree/'),
            'undirected_graph': Path('images/undirected_graph/'),
            'directed_graph': Path('images/directed_graph/')
        }
        self.yaml_path = Path('data/')
        self.text_path = Path('text/')
        self.results_path = Path('results/')
        self.create_directories()

    def create_directories(self):
        for path in self.image_path.values():
            path.mkdir(parents=True, exist_ok=True)
        self.yaml_path.mkdir(parents=True, exist_ok=True)
        self.text_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)

class BatchGeneration:
    def __init__(self, paths):
        self.paths = paths
        self.batch_generator = BatchGenerator()
        
        self.class_to_path = { # internal mapping from structure classes to path keys
            BinaryTree: 'binary_tree',
            BinarySearchTree: 'binary_search_tree',
            UndirectedGraph: 'undirected_graph',
            DirectedGraph: 'directed_graph',
        }

    async def generate_batches(self, selected_structure_classes, generations=1, variations=1):
        for structure_class in selected_structure_classes:
            path_key = self.class_to_path[structure_class]
            await self.batch_generator.generate_batch(
                structure_class=structure_class,
                type=structure_class.__name__.lower()[:2],
                yaml_name=f'{path_key}.yaml',
                yaml_path=self.paths.yaml_path,
                save_path=self.paths.image_path[path_key],
                text_path=self.paths.text_path,
                text_name=f'{path_key}_text.yaml',
                generations=generations,
                variations=variations,
            )

class Evaluation:
    def __init__(self, api_keys, paths):
        self.models = {
            'openai': OpenAI(api_key=api_keys['openai']),
            'deepmind': DeepMind(api_key=api_keys['deepmind'])
        }
        self.paths = paths

    async def evaluate_models(self, selected_structure_names):
        evaluator = Evaluator()
        for model_name, structure_name in selected_structure_names:
            model = self.models[model_name]
            messages = self.prepare_messages(model_name)

            yaml_name = f'{structure_name}.yaml'
            csv_name = f'{model_name}_{structure_name}.csv'
            await evaluator.evaluate(
                model=model,
                messages=messages,
                yaml_path=self.paths.yaml_path,
                yaml_name=yaml_name,
                csv_path=self.paths.results_path,
                csv_name=csv_name
            )

    def prepare_messages(self, model_name):
        if model_name == 'openai':
            return [
                SystemMessage(content="You are GPT-4V. A large multimodal model with vision capabilities trained by OpenAI."),
                UserMessage(content="{{content}}", images=["{{image}}"]),
            ]
        elif model_name == 'deepmind':
            return [
                BaseMessage(content="{{content}}"),
                ImageMessage(image="{{image}}"),
            ]

if __name__ == "__main__":
    main = Main()
    asyncio.run(main.run())