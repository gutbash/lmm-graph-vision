"""Contains the OpenAI model class for running completions."""

import openai
from time import perf_counter
from pathlib import Path
from typing import List, Dict, Literal, TypeVar

from evaluation.models.messages.message import UserMessage, SystemMessage, AssistantMessage, Roles
from utils.logger import Logger
from utils.encoder import encode_image

logger = Logger(__name__)

Messages = TypeVar("Messages", UserMessage, SystemMessage, AssistantMessage)

class OpenAI:
    """
    OpenAI model for running completions.
    
    Attributes
    ----------
    client : openai.OpenAI
        the OpenAI client
    api_key : str
        the API key for the OpenAI client
    model : str
        the model to use
    frequency_penalty : float
        the frequency penalty
    presence_penalty : float
        the presence penalty
    logit_bias : Dict[str, int]
        the logit bias values
    logprobs : bool
        whether to return the logprobs
    top_logprobs : int
        the number of top logprobs to return
    max_tokens : int
        the maximum number of tokens
    n : int
        the number of completions to generate
    seed : int
        the seed
    stop : List[str]
        the stop sequence
    temperature : float
        the temperature
    top_p : float
        the top p value
        
    Example
    -------
    ```python
    from evaluation.models.openai import OpenAI
    openai = OpenAI(api_key)
    ```
    """
    client: openai.OpenAI
    api_key: str
    model: str = "gpt-4-vision-preview"
    frequency_penalty: float = 0.0 # -2.0 to 2.0
    presence_penalty: float = 0.0 # -2.0 to 2.0
    logit_bias: Dict[str, int] = None # -100 to 100
    logprobs: bool = False # True or False
    top_logprobs: int = None # 0 to 5, logprobs must be True
    max_tokens: int = 500 # 1 to 4096
    n: int = None
    seed: int = None
    stop: List[str] = None
    temperature: float = 1.0 # 0.0 to 2.0
    top_p: float = 1.0 # 0.0 to 1.0
    
    def __init__(self, api_key: str, frequency_penalty: float = 0.0, presence_penalty: float = 0.0, logit_bias: Dict[str, int] = None, logprobs: bool = False, top_logprobs: int = None, max_tokens: int = 500, n: int = None, seed: int = None, stop: List[str] = None, temperature: float = 1.0, top_p: float = 1.0) -> None:
        """
        Initialize the OpenAI model.
        
        Parameters
        ----------
        api_key : str
            the API key for the OpenAI client
        frequency_penalty : float
            the frequency penalty
        presence_penalty : float
            the presence penalty
        logit_bias : Dict[str, int]
            the logit bias values
        logprobs : bool
            whether to return the logprobs
        top_logprobs : int
            the number of top logprobs to return
        max_tokens : int
            the maximum number of tokens
        n : int
            the number of completions to generate
        seed : int
            the seed
        stop : List[str]
            the stop sequence
        temperature : float
            the temperature
        top_p : float
            the top p value
        """
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logit_bias = logit_bias
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.max_tokens = max_tokens
        self.n = n
        self.seed = seed
        self.stop = stop
        self.temperature = temperature
        self.top_p = top_p
        
    def run(self, messages: List[Messages]) -> str:
        """
        Run the OpenAI completion.
        
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
        from evaluation.models.openai import OpenAI
        from evaluation.models.messages.message import UserMessage, AssistantMessage
        
        openai = OpenAI(api_key)
        messages = [
            UserMessage("Hello, how are you?"),
            AssistantMessage("I'm good, how are you?")
        ]
        
        completion = openai.run(messages)
        ```
        """
        
        try:
            
            messages = [message.to_message() for message in messages]
            
            logger.info(f"Running OpenAI Completion...")
            logger.info(messages)
            start = perf_counter()
        
            completion = self.client.with_options(max_retries=5).chat.completions.create(
                messages=messages,
                model=self.model,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                max_tokens=self.max_tokens,
                seed=self.seed,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            end = perf_counter()
        
        except Exception as e:
            logger.error(f'{type(e).__name__} @ {__name__}: {e}')
            return
        
        id = completion.id
        created = completion.created
        fingerprint = completion.system_fingerprint
        
        choices = completion.choices
        content = choices[0].message.content
        logprobs = choices[0].logprobs
        finish_reason = choices[0].finish_reason
        
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        total_tokens = completion.usage.total_tokens
        
        logger.info(f"│\n         │\n{content}\n         │")
        logger.info(f"│   {created} {id} {fingerprint} {finish_reason} {prompt_tokens} {completion_tokens} {total_tokens}\n         │")
        logger.info(f"╰── Ran OpenAI Completion in {round(end - start, 2)} seconds.")
        
        return content