"""Contains the DeepMind model class for running completions."""

import google.generativeai as deepmind
from utils.logger import Logger
from time import perf_counter
import traceback
import httpx
import json
import asyncio

from evaluation.models.messages.message import UserMessage, ModelMessage
from typing import List, TypeVar

logger = Logger(__name__)

Messages = TypeVar("Messages", UserMessage, ModelMessage)

class DeepMind:
    """
    DeepMind model for running completions.
    
    Attributes
    ----------
    api_key : str
        the API key for the DeepMind client
    client : deepmind.GenerativeModel
        the DeepMind client
    model : str
        the model to use
        
    Example
    -------
    ```python
    from evaluation.models.deepmind import DeepMind
    deepmind = DeepMind(api_key)
    ```
    """
    
    api_key: str
    client: deepmind.GenerativeModel
    model: str = 'gemini-pro-vision'
    
    def __init__(self, api_key: str) -> None:
        """
        Initializes the DeepMind model.
        
        Parameters
        ----------
        api_key : str
            the API key for the DeepMind client
        """
        self.api_key = api_key
        self.client = deepmind.GenerativeModel(self.model)
        
        deepmind.configure(api_key=api_key)
        
    def run(self, messages: List[Messages]) -> str:
        """
        Runs the DeepMind completion.
        
        Parameters
        ----------
        messages : List[Messages]
            the messages to run the completion on
        
        Returns
        -------
        str
            the completion
            
        Raises
        ------
        Exception
            if the completion fails
            
        Example
        -------
        ```python
        from evaluation.models.deepmind import DeepMind
        from evaluation.models.messages.message import BaseMessage, ImageMessage
        from pathlib import Path
        
        deepmind = DeepMind(api_key)
        messages = [
            BaseMessage("Hello, how are you?"),
            ImageMessage(Path("image.jpg"))
        ]
        
        completion = deepmind.run(messages)
        ```
        """
        
        try:
            
            messages = [message.to_deepmind() for message in messages]
            
            logger.info(f"Running DeepMind Completion...")
            logger.info(str(messages)[:200])
            start = perf_counter()
            
            completion = self.client.generate_content(contents=messages)
            
            end = perf_counter()
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f'{type(e).__name__} @ {__name__}: {e}\n{tb}')
            return
        
        content = completion.text
        
        logger.info(f"│\n         │\n{content}\n         │")
        logger.info(f"╰── Ran DeepMind Completion in {round(end - start, 2)} seconds.")
        
        return content
    
    async def arun(self, messages: List[Messages]) -> str:
        
        messages = [message.to_deepmind() for message in messages]
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {"contents": [messages]}
        
        timeout = httpx.Timeout(9999.0, connect=60.0)
        
        max_retries = 10
        retry_delay_yellow = 10
        retry_delay_red = 60
        
        for attempt in range(max_retries):
            try:
                
                logger.info(f"[A{attempt+1}] Running DeepMind Completion...")
                logger.info(str(messages)[:200])
                start = perf_counter()
                
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key={self.api_key}', headers=headers, data=json.dumps(data))
                    completion = response.json()
                    
                end = perf_counter()
                    
                if response.status_code == 500:
                    if attempt == max_retries - 1:
                        logger.error(f"500 Internal Server Error")
                        #logger.error(f"Retrying in {retry_delay_red} seconds...")
                        #await asyncio.sleep(retry_delay_red)
                    else:
                        logger.error(f"500 Internal Server Error")
                        #logger.error(f"Retrying in {retry_delay_yellow} seconds...")
                        #await asyncio.sleep(retry_delay_yellow)
                    continue
                if response.status_code == 429:
                    logger.error(f"429 Resource Exhausted")
                    #logger.error(f"Retrying in {retry_delay_red} seconds...")
                    #await asyncio.sleep(retry_delay_red)
                    continue
                
                content = completion['candidates'][0]['content']['parts'][0]['text']
                
                logger.info(f"│\n         │\n{content}\n         │")
                logger.info(f"╰── Ran DeepMind Completion in {round(end - start, 2)} seconds.")
                
                return content
                
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f'{type(e).__name__} @ {__name__}: {e}\n{tb}')
                
                if attempt < max_retries - 1:
                    #logger.error(f"Retrying in {retry_delay_red} seconds...")
                    #await asyncio.sleep(retry_delay_red)
                    pass
                else:
                    logger.error(f"Failed to run DeepMind Completion after {max_retries} attempts.")
                    return None
        return None