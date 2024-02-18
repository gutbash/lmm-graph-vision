from typing import List, Literal, Union
from pathlib import Path
from PIL import Image
from utils.logger import Logger

logger = Logger(__name__)

from utils.encoder import encode_image

Roles = Literal["user", "assistant", "system"]

class UserMessage:
    
    role: Roles = "user"
    content: str = None
    image_urls: List[Union[Path, str]] = None
    
    def __init__(self, content: str = None, image_urls: List[Union[Path, str]] = None) -> None:
        self.content = content
        self.image_urls = image_urls
        
        for image in self.image_urls:
            if isinstance(image, str) and image != "{{image}}":
                logger.error("Image must be be either a Path or a string with the placerholder '{{image}}'.")
        
    def to_message(self) -> dict:
        content_list = [{"type": "text", "text": f"{self.content}"}] if self.content else []
        images_list = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_url)}"}}
            for image_url in self.image_urls
        ] if self.image_urls else []

        return {
            "role": "user",
            "content": content_list + images_list
        }
        
class SystemMessage:
    
    role: Roles = "system"
    content: str
    
    def __init__(self, content: str) -> None:
        self.content = content
    
    def to_message(self) -> dict:
        return {
            "role": "system",
            "content": f"{self.content}"
        }
        
class AssistantMessage:
    
    role: Roles = "assistant"
    content: str
    
    def __init__(self, content: str) -> None:
        self.content = content
    
    def to_message(self) -> dict:
        return {
            "role": "assistant",
            "content": f"{self.content}"
        }
        
class ImageMessage:
    
    image: Union[Path, str]
    
    def __init__(self, image: Union[Path, str]) -> None:
        
        if isinstance(image, str) and image != "{{image}}":
            logger.error("Image must be be either a Path or a string with the placerholder '{{image}}'.")
        
        self.image = image if image == "{{image}}" else Image.open(image)
    
    def to_message(self) -> Union[str, Image.Image]:
        return self.image
        
class BaseMessage:
    
    content: str
    
    def __init__(self, content: str) -> None:
        self.content = content
    
    def to_message(self) -> str:
        return self.content