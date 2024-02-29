"""Contains classes for evaluating the performance of the models."""

from pathlib import Path
import pandas as pd
import yaml
from typing import TypeVar, List
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

logger = Logger(__name__)

from evaluation.models.openai import OpenAI
from evaluation.models.deepmind import DeepMind
from evaluation.models.messages.message import UserMessage, SystemMessage, AssistantMessage, ImageMessage, BaseMessage

Messages = TypeVar("Messages", UserMessage, SystemMessage, AssistantMessage, ImageMessage, BaseMessage)

Model = TypeVar('Model', OpenAI, DeepMind)

class Evaluator:
  """
  Evaluator for the models.
  
  Attributes
  ----------
  dataframe : pd.DataFrame
      the DataFrame for the evaluation
  columns : list
      the columns for the DataFrame
  """
  columns: list = ['run', 'n_generation', 'n_variation', 'n_format', 'structure', 'text_task', 'text_prompt', 'image_prompt', 'model_response', 'extracted_response', 'expected_response', 'match', 'similarity', 'node_font', 'node_color', 'edge_width', 'task_id', 'attempt', 'num_nodes', 'resolution']
  
  async def evaluate(self, model: Model, messages: List[Messages], yaml_path: Path, yaml_name: str, csv_path: Path, csv_name: str, repeats: int = 1) -> None:
    """
    Evaluates the model.
    
    Parameters
    ----------
    model : Model
        the model to evaluate
    messages : List[Messages]
        the messages to evaluate
    limit : int (optional = None)
        the limit for the evaluation
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
        
    Notes
    -----
    The evaluation YAML file has a default format for evaluation. This format is generated by the `BatchGenerator` and is as follows:
    
    ```yaml
    - color: '#88d7fe'
      expected: '[30, 71, 56, 69, 42, 19, 25]'
      font: serif
      format: 2
      generation: 1
      id: eab69c75-2644-496b-806e-fb6daa36347a
      image_path: images/binary_tree/bit_run-2_gen-1_var-1_fmt-2_thk-05_clr-88d7fe_fnt-serif_idn-eab69c75-2644-496b-806e-fb6daa36347a.png
      structure: binary_tree
      text: Provide a single-line python list representing the post-order traversal of
        the binary tree.
      width: '0.5'
      variation: 1
    ```
    """
    
    RATE_LIMIT = 60
    REQUEST_INTERVAL = 60
    
    save_path = validate_path(csv_path, csv_name, '.csv')
    file_exists = save_path.is_file()
    
    try:
      async with aiofiles.open(validate_path(yaml_path, yaml_name, '.yaml'), 'r') as file:
        prompts = yaml.safe_load(await file.read())
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
    
    chunk_size = RATE_LIMIT // repeats
    logger.info(f'Chunk size: {chunk_size}')
    if chunk_size == 0:
        chunk_size = 1
    
    prompt_chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]
    
    total_chunks = len(prompt_chunks)
    
    for chunk_index, chunk in enumerate(prompt_chunks):
      chunk_num = chunk_index + 1
      logger.info(f'Running {chunk_num} of {total_chunks} chunks')
    
      coroutines = []

      for i, prompt in enumerate(chunk):
        for attempt in range(repeats):
          message_list = copy.deepcopy(messages)

          image_path = Path(prompt.get('image_path'))
          
          for message in message_list:
            if hasattr(message, 'content'):
              if "{{content}}" in message.content:
                message.content = message.content.replace("{{content}}", prompt.get('text'))
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
                        
          coroutines.append((model.arun(message_list), prompt))
          
      logger.info(f'Running {len(coroutines)} coroutines...')

      try:
        results = await asyncio.gather(*[coro for coro, _ in coroutines])
      except Exception as e:
          tb = traceback.format_exc()
          logger.error(f'{type(e).__name__} @ {__name__}: {e}\n{tb}')
          return
    
      for result, prompt in zip(results, [prompt for _, prompt in coroutines]):
          content = result
          
          pattern = r'(\{.*?\})|(\[.*?\])'
          matches = re.findall(pattern, content)
          #logger.info(f'Matches: {matches}')
          clean_matches = [match for group in matches for match in group if match]
          if type(clean_matches[0]) is tuple:
            clean_matches = [item for tup in matches for item in tup if item]
          clean_match = str(clean_matches[0])
            
          express_expected = literal_eval(prompt.get('expected'))
          express_actual = literal_eval(clean_match)
          
          if type(express_expected) is list and type(express_actual) is list:
            similarity = calculate_similarity_list(express_expected, express_actual)
          elif type(express_expected) is dict and type(express_actual) is dict:
            similarity = calculate_similarity_dict(express_expected, express_actual)
          else:
            logger.error(f'Expected and actual types do not match: {type(express_expected)} and {type(express_actual)}')
            return
          
          # Append new data to the DataFrame
          new_row = {
            'run': prompt.get('run'),
            'n_generation': prompt.get('generation'),
            'n_variation': prompt.get('variation'),
            'n_format': prompt.get('format'),
            'structure': prompt.get('structure'),
            'text_task': str(prompt.get('text')),
            'text_prompt': str([message.content for message in message_list if type(message) == UserMessage or type(message) == BaseMessage]).strip("]["),
            'image_prompt': image_path,
            'model_response': content.replace('\n', '\\n'),
            'extracted_response': clean_match,
            'expected_response': prompt.get('expected'),
            'match': True if similarity >= 100.0 else False,
            'similarity': similarity,
            'node_font': prompt.get('font'),
            'node_color': prompt.get('color'),
            'edge_width': prompt.get('width'),
            'task_id': prompt.get('id'),
            'attempt': attempt + 1,
            'num_nodes': prompt.get('num_nodes'),
            'resolution': prompt.get('resolution'),
          }
          
          new_rows.append(new_row)
          
      # Sleep after processing each chunk to respect the rate limit
      logger.info(f'Completed {chunk_num} of {total_chunks} chunks')
      if chunk_num < total_chunks:  # Avoid sleeping after the last chunk
        logger.info(f'Sleeping for {REQUEST_INTERVAL} seconds to respect rate limit...')
        await asyncio.sleep(REQUEST_INTERVAL)
      
    async with aiofiles.open(save_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for new_row in new_rows:
            await writer.writerow([str(new_row[col]) for col in self.columns])