from pathlib import Path
import pandas as pd
import yaml
from typing import TypeVar
from utils.logger import Logger

logger = Logger(__name__)

from evaluation.models.openai import OpenAI
from evaluation.models.deepmind import DeepMind

yaml_path_binary_tree = Path('data/develop/binary_tree/')
yaml_path_binary_search_tree = Path('data/develop/binary_search_tree/')
yaml_path_undirected_graph = Path('data/develop/undirected_graph/')
yaml_path_directed_graph = Path('data/develop/directed_graph/')

class Evaluator:

  columns: list = ['generation', 'variation', 'format', 'structure', 'prompt', 'response', 'expected', 'match', 'image_path', 'font', 'color', 'thickness', 'task_id']
  rows: list = []
  dataframe: pd.DataFrame = None
  
  Model = TypeVar('Model', OpenAI, DeepMind)
  
  def evaluate(self, model: Model, limit: int, yaml_path: Path, yaml_name: str, csv_path: Path, csv_name: str) -> None:
    
    self.rows.clear()
    
    save_path = Path.joinpath(csv_path, csv_name)
    
    with open(Path.joinpath(yaml_path, yaml_name), 'r') as file:
      prompts = yaml.safe_load(file) or []

    # Check if the file exists, if not, create a new one with headers
    try:
      self.dataframe = pd.read_csv(save_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
      self.dataframe = pd.DataFrame(columns=self.columns)
      self.dataframe.to_csv(save_path, index=False)

    for i, prompt in enumerate(prompts):
      if i >= limit:
        break
      else:
        try:
          # Path to your image
          image_path = Path(prompt.get('image_path'))
          
          content = model.run(prompt.get('text'), image_path)
          
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