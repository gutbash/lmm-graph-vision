from pathlib import Path
import pandas as pd
import yaml
from typing import TypeVar, List
from utils.logger import Logger

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
  rows: list = []
  dataframe: pd.DataFrame = pd.DataFrame(columns=columns)
  
  model: Model
  limit: int
  yaml_path: Path
  yaml_name: str
  csv_path: Path
  csv_name: str
  
  def evaluate(self, model: Model, messages: List[Messages], limit: int, yaml_path: Path, yaml_name: str, csv_path: Path, csv_name: str) -> None:
    
    self.rows.clear()
    
    save_path = Path.joinpath(csv_path, csv_name)
    
    with open(Path.joinpath(yaml_path, yaml_name), 'r') as file:
      prompts = yaml.safe_load(file) or []

    # Check if the file exists, if not, create a new one with headers
    try:
      self.dataframe = pd.read_csv(save_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
      self.dataframe.to_csv(save_path, index=False)

    for i, prompt in enumerate(prompts):
      if i >= limit:
        break
      else:
        try:
          # Path to your image
          
          image_path = Path(prompt.get('image_path'))
          
          for message in messages:
              if hasattr(message, 'image_urls') and message.image_urls:
                  images = []
                  for image in message.image_urls:
                      if image == "{{image}}":
                          image = image_path
                      images.append(image)
                  message.image_urls = images
              elif hasattr(message, 'image') and message.image:
                  if message.image == "{{image}}":
                      message.image = Image.open(image_path)

          for message in messages:
            if hasattr(message, 'content') and message.content:
              if "{{content}}" in message.content:
                message.content = message.content.replace("{{content}}", prompt.get('text'))
              
          content = model.run(messages)
          
          if prompt.get('expected').strip("][}{") in content:
            match = True
          else:
            match = False

          # Append new data to the DataFrame
          new_row = pd.Series([
            prompt.get('generation'),
            prompt.get('variation'),
            prompt.get('format'),
            prompt.get('structure'),
            str(prompt.get('text')),
            content.replace('\n', '\\n'),
            prompt.get('expected'),
            match,
            image_path,
            prompt.get('font'),
            prompt.get('color'),
            prompt.get('thickness'),
            prompt.get('id'),
          ], index=self.columns)
          self.rows.append(new_row)

        except Exception as e:
          logger.error(f'{type(e).__name__} @ {__name__}: {e}')
    
    self.dataframe = pd.concat([self.dataframe, pd.DataFrame(self.rows, columns=self.columns)], ignore_index=True)

    # Write the updated DataFrame back to the CSV file
    self.dataframe.to_csv(save_path, index=False)