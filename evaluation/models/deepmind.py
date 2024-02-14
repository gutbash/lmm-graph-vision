import google.generativeai as deepmind
from pathlib import Path
import PIL.Image as Image
from utils.logger import Logger
from time import perf_counter

logger = Logger(__name__)

class DeepMind:
    
    api_key: str
    client: deepmind.GenerativeModel
    model: str = 'gemini-pro-vision'
    
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = deepmind.GenerativeModel(self.model)
        
        deepmind.configure(api_key=api_key)
        
    def run(self, text: str, image_path: Path) -> str:
        
        image = Image.open(image_path)
        
        try:
            
            logger.info(f"Running DeepMind Completion...")
            start = perf_counter()
            
            completion = self.client.generate_content([text, image])
            
            end = perf_counter()
            
        except Exception as e:
            logger.error(e)
            return None
        
        
        content = completion.text
        
        logger.info(f"│\n         │\n{content}\n         │")
        logger.info(f"╰── Ran DeepMind Completion in {round(end - start, 2)} seconds.")
        
        return content