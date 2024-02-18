from pathlib import Path
import pandas as pd
import yaml
from typing import TypeVar, List
from utils.logger import Logger
import copy

from utils.encoder import encode_image
from PIL import Image

logger = Logger(__name__)

from evaluation.models.openai import OpenAI
from evaluation.models.deepmind import DeepMind
from evaluation.models.messages.message import UserMessage, SystemMessage, AssistantMessage, ImageMessage, BaseMessage

Messages = TypeVar("Messages", UserMessage, SystemMessage, AssistantMessage, ImageMessage, BaseMessage)

yaml_path_binary_tree = Path('data/develop/binary_tree/')
yaml_path_binary_search_tree = Path('data/develop/binary_search_tree/')
yaml_path_undirected_graph = Path('data/develop/undirected_graph/')
yaml_path_directed_graph = Path('data/develop/directed_graph/')

Model = TypeVar('Model', OpenAI, DeepMind)

class Evaluator:

  columns: list = ['generation', 'variation', 'format', 'structure', 'prompt', 'response', 'expected', 'match', 'image_path', 'font', 'color', 'thickness', 'task_id']
  
  def evaluate(self, model: Model, messages: List[Messages], limit: int, yaml_path: Path, yaml_name: str, csv_path: Path, csv_name: str) -> None:
    
    save_path = Path.joinpath(csv_path, csv_name)
    file_exists = save_path.is_file()
    
    with open(Path.joinpath(yaml_path, yaml_name), 'r') as file:
      prompts = yaml.safe_load(file) or []

    # Check if the file exists, if not, create a new one with headers
    try:
      self.dataframe = pd.read_csv(save_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
      logger.info(f'Creating new CSV at {save_path}')
      self.dataframe = pd.DataFrame(columns=self.columns)
      self.dataframe.to_csv(save_path, index=False)

    for i, prompt in enumerate(prompts):
      message_list = copy.deepcopy(messages)
      
      if i >= limit:
        break
      else:
        try:
          # Path to your image
          
          image_path = Path(prompt.get('image_path'))
          
          for message in message_list:
              if hasattr(message, 'content'):
                if "{{content}}" in message.content:
                  message.content = message.content.replace("{{content}}", prompt.get('text'))
              if hasattr(message, 'image_urls'):
                  images = []
                  for image in message.image_urls:
                      if image == "{{image}}":
                          image = image_path
                      images.append(image)
                  message.image_urls = images
              elif hasattr(message, 'image'):
                  if message.image == "{{image}}":
                      message.image = Image.open(image_path)

          content = model.run(message_list)
          
          if prompt.get('expected').strip("][}{") in content:
            match = True
          else:
            match = False

          # Append new data to the DataFrame
          new_row = {
            'generation': prompt.get('generation'),
            'variation': prompt.get('variation'),
            'format': prompt.get('format'),
            'structure': prompt.get('structure'),
            'prompt': str(prompt.get('text')),
            'response': content.replace('\n', '\\n'),
            'expected': prompt.get('expected'),
            'match': match,
            'image_path': image_path,
            'font': prompt.get('font'),
            'color': prompt.get('color'),
            'thickness': prompt.get('thickness'),
            'task_id': prompt.get('id'),
          }
          
          pd.DataFrame([new_row]).to_csv(save_path, mode='a', header=not file_exists, index=False)
          file_exists = True  # After the first write, header should not be written again.

        except Exception as e:
          logger.error(f'{type(e).__name__} @ {__name__}: {e}')