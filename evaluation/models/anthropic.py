"""Contains the Anthropic model class for running completions."""

from utils.logger import Logger
from time import perf_counter
import traceback
import anthropic
import asyncio

from evaluation.models.messages.message import UserMessage, ModelMessage
from typing import List, TypeVar, Literal

logger = Logger(__name__)

Messages = TypeVar("Messages", UserMessage, ModelMessage)
Model = Literal['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']

class Anthropic:
    """
    Anthropic model for running completions.
    
    Attributes
    ----------
    api_key : str
        the API key for the Anthropic client
    client : anthropic.Anthropic
        the Anthropic client
    model : str
        the model to use
    """
    
    api_key: str
    client: anthropic.Anthropic
    model: str = 'claude-3-opus-20240229'
    
    def __init__(self, api_key: str, model: Model = 'claude-3-opus-20240229') -> None:
        """
        Initializes the DeepMind model.
        
        Parameters
        ----------
        api_key : str
            the API key for the DeepMind client
        """
        self.api_key = api_key
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
    
    async def arun(self, messages: List[Messages]) -> str:
        
        messages = [message.to_anthropic() for message in messages]
        
        max_retries = 1000
        for attempt in range(max_retries):
            try:
                
                logger.info(f"[A{attempt+1}] Running Anthropic Completion...")
                logger.info(str(messages)[:200])
                start = perf_counter()

                completion = await self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=messages
                )
                    
                end = perf_counter()
                
            except Exception as e:
                tb = traceback.format_exc()
                #logger.error(f'{type(e).__name__} @ {__name__}: {e}\n{tb}')
                logger.error(f"429 Resource Exhausted")
                await asyncio.sleep(1)
                
                if attempt < max_retries - 1:
                    #logger.error(f"Retrying in {retry_delay_red} seconds...")
                    #await asyncio.sleep(retry_delay_red)
                    continue
                else:
                    logger.error(f"Failed to run Anthropic Completion after {max_retries} attempts.")
                    return None
                
            content = completion.content[0].text
            
            logger.info(f"│\n         │\n{content}\n         │")
            logger.info(f"╰── Ran Anthropic Completion in {round(end - start, 2)} seconds.")
            
            return content
        
        return None