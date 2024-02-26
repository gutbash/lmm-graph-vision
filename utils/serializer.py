"""Contains functions for performing operations on YAML files."""

import yaml
from uuid import UUID
from pathlib import Path

def add_object(uuid: UUID = None, file_path: Path = None, text: str = None, image_path: Path = None, expected: str = None, structure: str = None, run: int = 0, generation: int = None, variation: int = None, format: int = None, color: str = None, font: str = None, width: str = None, num_nodes: int = None, resolution: int = None) -> None:
    """
    Add an object to a YAML file.
        
    Paramters
    ---------
    uuid : UUID (default: None)
        the UUID of the object
    file_path : Path (default: None)
        the path to the YAML file
    text : str (default: None)
        the text prompt for the model
    image_path : Path (default: None)
        the path to the image
    expected : str (default: None)
        the expected output of the model
    structure : str (default: None)
        the data structure
    run : int (default: 0)
        the run number
    generation : int (default: None)
        the generation number
    variation : int (default: None)
        the variation number
    format : int (default: None)
        the format number
    color : str (default: None)
        the color of the text
    font : str (default: None)
        the font of the text
    width : str (default: None)
        the width of the edges
    
    Returns
    -------
    None
    
    Raises
    ------
    Exception
        if no file path is provided
        if the file path does not exist
        if no image path is provided
        if the image path does not exist
    """
    
    if file_path is None:
        raise Exception('No file path provided.')
    elif not file_path.exists():
        raise Exception('The file path does not exist.')
    
    if image_path is None:
        raise Exception('No image path provided.')
    elif not image_path.exists():
        raise Exception('The image path does not exist.')
    
    new_object = {
        'id': str(uuid),
        'text': text,
        'image_path': str(image_path),
        'expected': expected,
        'structure': structure,
        'generation': generation,
        'variation': variation,
        'format': format,
        'color': color,
        'font': font,
        'width': width,
        'num_nodes': num_nodes,
        'resolution': resolution,
        'run': run,
    }
    
    try:
        # Read the existing data from the file
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file) or []

        # Append the new object
        data.append(new_object)

        # Write the updated data back to the file
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file)

    except Exception as e:
        print(f"An error occurred: {e}")