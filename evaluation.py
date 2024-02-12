from pathlib import Path
import pandas as pd
import requests
import yaml

from utils.encoding import encode_image

yaml_path_binary_tree = Path('data/develop/binary_tree/')
yaml_path_binary_search_tree = Path('data/develop/binary_search_tree/')
yaml_path_undirected_graph = Path('data/develop/undirected_graph/')
yaml_path_directed_graph = Path('data/develop/directed_graph/')

class Evaluator:
  
  api_key: str
  columns: list = ['generation', 'variation', 'format', 'structure', 'prompt', 'response', 'image_path', 'font', 'color', 'thickness', 'task_id', 'api_id', 'created']
  limit: int
  path: Path
  filename: str
  rows: list = []
  dataframe: pd.DataFrame
  
  def __init__(self, limit: int, path: Path, filename: str, api_key: str) -> None:
    self.api_key = api_key
    self.limit = limit
    self.path = path
    self.filename = filename
  
  def evaluate(self) -> None:
    
    with open(Path.joinpath(self.path, self.filename), 'r') as file:
      prompts = yaml.safe_load(file) or []

    # Check if the file exists, if not, create a new one with headers
    try:
      self.dataframe = pd.read_csv('results/results.csv')
    except (FileNotFoundError, pd.errors.EmptyDataError):
      self.dataframe = pd.DataFrame(columns=self.columns)
      self.dataframe.to_csv('results/results.csv', index=False)

    for i, prompt in enumerate(prompts):
      if i >= self.limit:
        break
      else:
        try:
          # Path to your image
          image_path = Path(prompt.get('image_path'))

          # Getting the base64 string
          image_data = encode_image(image_path)
          headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
          }

          payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": f"{prompt.get('text')}"
                  },
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": f"data:image/jpeg;base64,{image_data}"
                    }
                  }
                ]
              }
            ],
            "max_tokens": 300
          }

          # Send the request to the API
          resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
          print(resp)
          response = resp.get('choices')[0].get('message').get('content').replace('\n', '\\n')

          # Append new data to the DataFrame
          new_row = pd.Series([
            prompt.get('generation'),
            prompt.get('variation'),
            prompt.get('format'),
            prompt.get('structure'),
            str(prompt.get('text')),
            response,
            image_path,
            prompt.get('font'),
            prompt.get('color'),
            prompt.get('thickness'),
            prompt.get('id'),
            resp.get('id'),
            resp.get('created'),
          ], index=self.columns)
          self.rows.append(new_row)

        except Exception as e:
          print(f"An error occurred: {e.with_traceback()}")
    
    self.dataframe = pd.concat([self.dataframe, pd.DataFrame(self.rows, columns=self.columns)], ignore_index=True)

    # Write the updated DataFrame back to the CSV file
    self.dataframe.to_csv('results/results.csv', index=False)