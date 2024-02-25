from generation.generator import Generator, BatchGenerator
from generation.structures.tree import BinaryTree, BinarySearchTree
from generation.structures.graph import UndirectedGraph, DirectedGraph

from evaluation.evaluator import Evaluator
from evaluation.models.openai import OpenAI
from evaluation.models.deepmind import DeepMind

from evaluation.models.messages.message import UserMessage, SystemMessage, AssistantMessage, ImageMessage, BaseMessage

import os
from pathlib import Path
from dotenv import load_dotenv

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
multiplier = 1

batch_generator.generate_batch(
    structure_class=BinaryTree,
    type='bit',
    yaml_name='binary_tree.yaml',
    yaml_path=yaml_path,
    save_path=image_path_binary_tree,
    text_path=text_path,
    text_name='binary_tree_text.yaml',
    generations=multiplier,
    variations=multiplier,
)

batch_generator.generate_batch(
    structure_class=BinarySearchTree,
    type='bst',
    yaml_name='binary_search_tree.yaml',
    yaml_path=yaml_path,
    save_path=image_path_binary_search_tree,
    text_path=text_path,
    text_name='binary_search_tree_text.yaml',
    generations=multiplier,
    variations=multiplier,
)

batch_generator.generate_batch(
    structure_class=UndirectedGraph,
    type='ug',
    yaml_name='undirected_graph.yaml',
    yaml_path=yaml_path,
    save_path=image_path_undirected_graph,
    text_path=text_path,
    text_name='undirected_graph_text.yaml',
    generations=multiplier,
    variations=multiplier,
)

batch_generator.generate_batch(
    structure_class=DirectedGraph,
    type='dg',
    yaml_name='directed_graph.yaml',
    yaml_path=yaml_path,
    save_path=image_path_directed_graph,
    text_path=text_path,
    text_name='directed_graph_text.yaml',
    generations=multiplier,
    variations=multiplier,
)

### TEST EVALUATION ###

openai = OpenAI(api_key=openai_api_key)
deepmind = DeepMind(api_key=deepmind_api_key)

openai_messages = [
    UserMessage(content="{{content}}", images=["{{image}}"]),
]

openai_sys_messages = [
    SystemMessage(content="You are GPT-4V. A large multimodal model with vision capabilities trained by OpenAI."),
    UserMessage(content="{{content}}", images=["{{image}}"]),
]

deepmind_messages = [
    BaseMessage(content="{{content}}"),
    ImageMessage(image="{{image}}"),
]

openai_csv = 'openai.csv'
deepmind_csv = 'deepmind.csv'

evaluator = Evaluator()

model = openai
csv_name = deepmind_csv
messages = deepmind_messages

#TODO: run eval on sys vs no sys
#TODO: run eval on str name vs no str name
#TODO: run eval on meta-inst vs no meta-inst
#TODO: run eval on large vs small

#evaluator.evaluate(model=model, messages=openai_messages, limit=None, yaml_path=yaml_path, yaml_name='binary_tree.yaml', csv_path=Path('results/'), csv_name="results_sys.csv")
#evaluator.evaluate(model=model, messages=openai_sys_messages, limit=None, yaml_path=yaml_path, yaml_name='binary_tree.yaml', csv_path=Path('results/'), csv_name="results_no_sys.csv")
#evaluator.evaluate(model=model, messages=messages, limit=3, yaml_path=yaml_path, yaml_name='binary_search_tree.yaml', csv_path=Path('results/'), csv_name=csv_name)
#evaluator.evaluate(model=model, messages=messages, limit=3, yaml_path=yaml_path, yaml_name='undirected_graph.yaml', csv_path=Path('results/'), csv_name=csv_name)
#evaluator.evaluate(model=model, messages=messages, limit=3, yaml_path=yaml_path, yaml_name='directed_graph.yaml', csv_path=Path('results/'), csv_name=csv_name)