from openai import OpenAI
import base64
from pathlib import Path
import os
import pandas as pd
import re
import csv
import json
import requests
import yaml

api_key=os.environ.get('OPENAI_API_KEY')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

folder_path = Path('./images')
file_list = [file.resolve() for file in folder_path.iterdir() if file.is_file()]

print(len(file_list))

for i in file_list:
  print(i)

limit = 1

columns = ['', 'id', 'created', 'model', 'content', 'file']

# Check if the file exists, if not, create a new one with headers
try:
    df = pd.read_csv('./data.csv')
except FileNotFoundError:
    df = pd.DataFrame(columns=columns)
    df.to_csv('./data.csv', index=False)

for i, path in enumerate(file_list):
    if i >= limit:
        break
    else:
        try:
            # Path to your image
            image_path = path

            # Getting the base64 string
            base64_image = encode_image(image_path)
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer {api_key}"
            }

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                  {
                    "role": "user",
                    "content": [
                      {
                        "type": "text",
                        "text": "You are a very experienced computer scientist who is an expert at answering question about graphs or trees. You share your reasoning process and you acknowledge when the problem can not be solved. In the provided diagram, identify the graph or tree and solve the problem that is described in box with the red text outline. Please explicitly say 'the answer is...' when providing the final answer. Be concise."
                      },
                      {
                        "type": "image_url",
                        "image_url": {
                          "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                      }
                    ]
                  }
                ],
                "max_tokens": 300
            }

            # Send the request to the API
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            resp = response.json()
            print(resp)
            content = resp.get('choices')[0].get('message').get('content')

            # Append new data to the DataFrame
            new_row = pd.Series([i, resp.get('id'), resp.get('created'), resp.get('model'), content, path], index=columns)
            df = df.append(new_row, ignore_index=True)

        except Exception as e:
            print(f"An error occurred: {e}")

# Write the updated DataFrame back to the CSV file
df.to_csv('./data.csv', index=False)

df.content