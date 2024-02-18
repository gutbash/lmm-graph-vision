import google.generativeai as deepmind
from utils.logger import Logger
from time import perf_counter
import traceback

from evaluation.models.messages.message import BaseMessage, ImageMessage
from typing import List, TypeVar

logger = Logger(__name__)

Messages = TypeVar("Messages", BaseMessage, ImageMessage)

class DeepMind:
    
    api_key: str
    client: deepmind.GenerativeModel
    model: str = 'gemini-pro-vision'
    
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = deepmind.GenerativeModel(self.model)
        
        # FIXME: client not recognizing api_key
        deepmind.configure(api_key=api_key)
        
    def run(self, messages: List[Messages]) -> str:
        
        try:
            
            messages = [message.to_message() for message in messages]
            
            logger.info(f"Running DeepMind Completion...")
            start = perf_counter()
            
            completion = self.client.generate_content(['hi'])
            
            end = perf_counter()
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f'{type(e).__name__} @ {__name__}: {e}\n{tb}')
            return
        
        content = completion.text
        
        logger.info(f"│\n         │\n{content}\n         │")
        logger.info(f"╰── Ran DeepMind Completion in {round(end - start, 2)} seconds.")
        
        return content