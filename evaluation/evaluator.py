"""Contains classes for evaluating the performance of the models."""

from pathlib import Path
import pandas as pd
import yaml
from typing import TypeVar, List, TypedDict, Literal
from utils.logger import Logger
import copy
from PIL import Image
import traceback
from utils.files import validate_path
import re
import aiofiles
import csv
import asyncio
from evaluation.similarity import calculate_similarity_list, calculate_similarity_dict
from ast import literal_eval
import time
from asyncio import Lock

logger = Logger(__name__)

from evaluation.models.openai import OpenAI
from evaluation.models.deepmind import DeepMind
from evaluation.models.messages.message import UserMessage, SystemMessage, ModelMessage

Message = TypeVar("Message", UserMessage, SystemMessage, ModelMessage)
Task = Literal['post_order', 'pre_order', 'in_order', 'breadth_first_search', 'depth_first_search', 'adjacency_list']
class Prompt(TypedDict):
  messages: List[Message]
  task: Task
Model = TypeVar('Model', OpenAI, DeepMind)

class RateLimiter:
    def __init__(self, calls_per_second):
        self.calls_per_second = calls_per_second
        self.lock = Lock()
        self.last_call_time = 0

    async def acquire(self):
        async with self.lock:
            current_time = time.monotonic()
            elapsed_time = current_time - self.last_call_time

            if elapsed_time < 1 / self.calls_per_second:
                await asyncio.sleep(1 / self.calls_per_second - elapsed_time)

            self.last_call_time = time.monotonic()

class Evaluator:
  """
  Evaluator for the models.
  
  Attributes
  ----------
  dataframe : pd.DataFrame
      the DataFrame for the evaluation
  columns : list
      the columns for the DataFrame
  semaphore : asyncio.Semaphore
      the global semaphore for rate limiting
  """
  columns: list = ['run_id', 'generation_id', 'variation_id', 'format_id', 'attempt_id', 'structure', 'task', 'text_prompt', 'image_prompt', 'response', 'predicted', 'ground_truth', 'match', 'similarity', 'node_font', 'node_color', 'node_shape', 'edge_width', 'arrow_style', 'num_nodes', 'resolution', 'task_id', 'image_id']
  rate_limiter = RateLimiter(0.9)
  
  def __init__(self) -> None:
    self.total_calls = 0
    self.completed_calls = 0
  
  async def evaluate(self, model: Model, prompts: list[Prompt], yaml_path: Path, yaml_name: str, csv_path: Path, csv_name: str, repeats: int = 1) -> None:
    """
    Evaluates the model.
    
    Parameters
    ----------
    model : Model
        the model to evaluate
    prompts : List[Prompt]
        the messages to evaluate
    yaml_path : Path
        the path to the evaluation YAML file
    yaml_name : str
        the name of the evaluation YAML file
    csv_path : Path
        the path to the results CSV file
    csv_name : str
        the name of the results CSV file
        
    Raises
    ------
    FileNotFoundError
        if the YAML file is not found
    pd.errors.EmptyDataError
        if the CSV file is empty
    """
    if isinstance(model, OpenAI):
      self.rate_limiter.calls_per_second = 5.0 # based on tier 4 300,000 RPM + 150,000 TPM
    elif isinstance(model, DeepMind):
      self.rate_limiter.calls_per_second = 0.9
    
    save_path = validate_path(csv_path, csv_name, '.csv')
    file_exists = save_path.is_file()
    
    try:
      async with aiofiles.open(validate_path(yaml_path, yaml_name, '.yaml'), 'r') as file:
        tasks = yaml.safe_load(await file.read())
    except Exception as e:
      tb = traceback.format_exc()
      logger.error(f'{type(e).__name__} @ {__name__}: {e}\n{tb}')
      raise FileNotFoundError(f'YAML file {yaml_name} not found.')

    # Check if the file exists, if not, create a new one with headers
    if not file_exists:
        logger.info(f'Creating new CSV at {save_path}')
        self.dataframe = pd.DataFrame(columns=self.columns)
        async with aiofiles.open(save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            await writer.writerow(self.columns)
        
    new_rows = []
    
    self.total_calls += len(tasks) * repeats
    
    logger.info(f'Total calls: {self.total_calls}')
    
    async def process_task(task, attempt):
      await self.rate_limiter.acquire()
      message_list = None
      
      for prompt in prompts:
        if prompt.get("task") == task.get('task'):
          message_list = copy.deepcopy(prompt.get('messages'))
          break

      image_path = Path(task.get('image_path'))
      
      for message in message_list:
        if hasattr(message, 'content'):
          if "{{structure}}" in message.content:
            message.content = message.content.replace("{{structure}}", task.get('structure').replace('_', ' '))
        if hasattr(message, 'images'):
            images = []
            for image in message.images:
                if image == "{{image}}":
                    image = image_path
                images.append(image)
            message.images = images
        elif hasattr(message, 'image'):
            if message.image == "{{image}}":
                message.image = image_path
                    
      result = await model.arun(message_list)
      
      self.completed_calls += 1
      
      logger.info(f'[T{self.completed_calls}/{self.total_calls}] Complete')
      
      clean_match = ''
      similarity = 0.0
      
      pattern = r'(\{.*?\})|(\[.*?\])'
      content = re.sub(' +', ' ', result.replace('\n', ''))
      matches = re.findall(pattern, content)
      if matches == []:
        logger.error(f'No matches found in content: {content}')
      else:
        logger.info(f'Matches: {matches}')
        clean_matches = [match for group in matches for match in group if match]
        if type(clean_matches[-1]) is tuple:
          clean_matches = [item for tup in matches for item in tup if item]
        
        express_expected = literal_eval(task.get('ground_truth'))
        for match in reversed(clean_matches):
          if match != '[]' and match != '{}':
            try:
              clean_match = str(match)
              express_actual = literal_eval(clean_match)
              break
            except:
              pass
          
        if type(express_expected) is list and type(express_actual) is list:
          similarity = calculate_similarity_list(express_expected, express_actual)
        elif type(express_expected) is dict and type(express_actual) is dict:
          similarity = calculate_similarity_dict(express_expected, express_actual)
        else:
          logger.error(f'Expected and actual types do not match: {type(express_expected)} and {type(express_actual)}')
          logger.info(f'Expected: {express_expected}, Actual: {express_actual}')
          logger.info(f'Content: {content}')
          #raise Exception('Expected and actual types do not match')
      
      # Append new data to the list of new rows
      new_row = {
        'generation_id': task.get('generation_id'),
        'variation_id': task.get('variation_id'),
        'format_id': task.get('format_id'),
        'attempt_id': attempt + 1,
        'structure': task.get('structure'),
        'task': task.get('task'),
        'text_prompt': str([message.content for message in message_list if type(message) == UserMessage]).strip("]["),
        'image_prompt': image_path,
        'response': content.replace('\n', '\\n'),
        'predicted': clean_match,
        'ground_truth': task.get('ground_truth'),
        'match': True if similarity >= 100.0 else False,
        'similarity': similarity,
        'node_font': task.get('node_font'),
        'node_color': task.get('node_color'),
        'node_shape': task.get('node_shape'),
        'edge_width': task.get('edge_width'),
        'arrow_style': task.get('arrow_style'),
        'num_nodes': task.get('num_nodes'),
        'resolution': task.get('resolution'),
        'task_id': task.get('task_id'),
        'image_id': task.get('image_id')
      }
      
      new_rows.append(new_row)
        
    coroutines = []
    for task in tasks:
      for attempt in range(repeats):
        coroutines.append(process_task(task, attempt))
        
    await asyncio.gather(*coroutines)
      
    async with aiofiles.open(save_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for index, new_row in enumerate(new_rows, start=1):
            row_data = [str(index)] + [str(new_row[col]) for col in self.columns if col != 'run_id']
            await writer.writerow(row_data)