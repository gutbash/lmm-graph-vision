"""Contains functions to encode an image to base64."""

import base64
from pathlib import Path

# Function to encode the image
def encode_image(image_path: Path = None) -> str:
    """
    Encode an image to base64.
    
    Parameters
    ----------
    image_path : Path
        The path to the image file.
    
    Returns
    -------
    str
        The base64 encoded image.
        
    Raises
    ------
    Exception
        If no image path is provided.
        If the image path does not exist.
        
    Example
    -------
    
    ```python
    from pathlib import Path
    from utils.encoder import encode_image
    
    image_path = Path('path/to/image.jpg')
    encoded_image = encode_image(image_path)
    ```
    """
    if image_path is None:
        raise Exception('No image path provided.')
    elif not image_path.exists():
        raise Exception('The image path does not exist.')
      
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')