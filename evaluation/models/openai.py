import openai
from time import perf_counter
from pathlib import Path
from typing import List, Dict

from utils.logger import Logger
from utils.encoder import encode_image

logger = Logger(__name__)

class OpenAI:
    
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
        
    def run(self, text: str, image_path: Path) -> str:
        
        messages = [
            {
                "role": "user",
                "content": [
                {"type": "text", "text": f"{text}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
                ]
            }
        ]
        
        try:
                
            logger.info(f"Running OpenAI Completion...")
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
            logger.error(e)
            return None
        
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