import base64
from pathlib import Path

# Function to encode the image
def encode_image(image_path: Path = None) -> str:
    """
    Encode an image to base64.
    
    Parameters
    ----------
    image_path : Path (default: None)
        the path to the image
    
    Returns
    -------
    str
        the encoded image
    
    Raises
    ------
    Exception
        if no image path is provided
        if the image path does not exist
    """
    
    if image_path is None:
        raise Exception('No image path provided.')
    elif not image_path.exists():
        raise Exception('The image path does not exist.')
      
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')