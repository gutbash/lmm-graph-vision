"""Contains the DeepMind model class for running completions."""

import google.generativeai as deepmind
from utils.logger import Logger
from time import perf_counter
import traceback
import httpx
import json
import asyncio

from evaluation.models.messages.message import BaseMessage, ImageMessage
from typing import List, TypeVar

logger = Logger(__name__)

Messages = TypeVar("Messages", BaseMessage, ImageMessage)

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
            
            messages = [message.to_message() for message in messages]
            
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
        
        messages = [message.to_message() for message in messages]
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {"contents": [{"parts": messages}]}
        
        timeout = httpx.Timeout(60.0, connect=5.0)
        
        try:
            
            logger.info(f"Running DeepMind Completion...")
            logger.info(str(messages)[:200])
            start = perf_counter()
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key={self.api_key}', headers=headers, data=json.dumps(data))
                completion = response.json()
            
            end = perf_counter()
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f'{type(e).__name__} @ {__name__}: {e}\n{tb}')
            return
        
        logger.info(f"│\n         │\n{completion}\n         │")
        
        content = completion['candidates'][0]['content']['parts'][0]['text']
        
        logger.info(f"│\n         │\n{content}\n         │")
        logger.info(f"╰── Ran DeepMind Completion in {round(end - start, 2)} seconds.")
        
        return content