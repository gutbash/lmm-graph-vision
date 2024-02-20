"""Contains message classes for the OpenAI and DeepMind models."""

from typing import List, Literal, Union
from pathlib import Path
from PIL import Image
from utils.logger import Logger

logger = Logger(__name__)

from utils.encoder import encode_image

Roles = Literal["user", "assistant", "system"]

class UserMessage:
    """
    User message for the OpenAI model.
    
    Attributes
    ----------
    role : Roles
        the role of the message
    content : str
        the content of the message
    images : List[Union[Path, str]]
        the images of the message
    """
    
    role: Roles = "user"
    content: str = None
    images: List[Union[Path, str]] = None
    
    def __init__(self, content: str = None, images: List[Union[Path, str]] = None) -> None:
        """
        Initializes the UserMessage.
        
        Parameters
        ----------
        content : str
            the content of the message
        images : List[Union[Path, str]]
            the images of the message
        """
        self.content = content
        self.images = images
        
        for image in self.images:
            if isinstance(image, str) and image != "{{image}}":
                logger.error("Image must be be either a Path or a string with the placerholder '{{image}}'.")
        
    def to_message(self) -> dict:
        """
        Converts message to API format.
        
        Returns
        -------
        dict
            the message in API format
        """
        content_list = [{"type": "text", "text": f"{self.content}"}] if self.content else []
        images_list = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_url)}"}}
            for image_url in self.images
        ] if self.images else []

        return {
            "role": "user",
            "content": content_list + images_list
        }
        
class SystemMessage:
    """
    System message for the OpenAI model.
    
    Attributes
    ----------
    role : Roles
        the role of the message
    content : str
        the content of the message
    """
    
    role: Roles = "system"
    content: str
    
    def __init__(self, content: str) -> None:
        """
        Initializes the SystemMessage.
        
        Parameters
        ----------
        content : str
            the content of the message
        """
        self.content = content
    
    def to_message(self) -> dict:
        """
        Converts message to API format.
        
        Returns
        -------
        dict
            the message in API format
        """
        return {
            "role": "system",
            "content": f"{self.content}"
        }
        
class AssistantMessage:
    """
    Assistant message for the OpenAI model.
    
    Attributes
    ----------
    role : Roles
        the role of the message
    content : str
        the content of the message
    """
    role: Roles = "assistant"
    content: str
    
    def __init__(self, content: str) -> None:
        """
        Initializes the AssistantMessage.
        
        Parameters
        ----------
        content : str
            the content of the message
        """
        self.content = content
    
    def to_message(self) -> dict:
        """
        Converts message to API format.
        
        Returns
        -------
        dict
            the message in API format
        """
        return {
            "role": "assistant",
            "content": f"{self.content}"
        }
        
class ImageMessage:
    """
    Image message for the DeepMind model.
    
    Attributes
    ----------
    image : Union[Path, str]
        the image of the message
    """
    image: Union[Path, str]
    
    def __init__(self, image: Union[Path, str]) -> None:
        """
        Initializes the ImageMessage.
        
        Parameters
        ----------
        image : Union[Path, str]
            the image of the message
        """
        if isinstance(image, str) and image != "{{image}}":
            logger.error("Image must be be either a Path or a string with the placerholder '{{image}}'.")
        
        self.image = image if image == "{{image}}" else Image.open(image)
    
    def to_message(self) -> Union[str, Image.Image]:
        """
        Converts message to API format.
        
        Returns
        -------
        Union[str, Image.Image]
            the message in API format
        """
        return self.image
        
class BaseMessage:
    """
    Base message for the DeepMind model.
    
    Attributes
    ----------
    content : str
        the content of the message
    """
    content: str
    
    def __init__(self, content: str) -> None:
        """
        Initializes the BaseMessage.
        
        Parameters
        ----------
        content : str
            the content of the message
        """
        self.content = content
    
    def to_message(self) -> str:
        """
        Converts message to API format.
        
        Returns
        -------
        str
            the message in API format
        """
        return self.content