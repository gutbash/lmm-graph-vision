"""Contains functions for performing operations on YAML files."""

import yaml
from pathlib import Path
import aiofiles

async def add_objects_async(objects: list, file_path: Path) -> None:
    """
    Asynchronously add a batch of objects to a YAML file.

    Parameters
    ----------
    objects : list
        A list of dictionaries, each representing an object to be added. Each dictionary should have keys corresponding
        to parameters like uuid, text, image_path, etc.
    file_path : Path
        The path to the YAML file.
    
    Raises
    ------
    Exception
        if no file path is provided
        if the file path does not exist
    """
    if file_path is None:
        raise Exception('No file path provided.')
    elif not file_path.exists():
        raise Exception('The file path does not exist.')

    try:
        # Read the existing data from the file asynchronously
        async with aiofiles.open(file_path, 'r') as file:
            content = await file.read()
            data = yaml.safe_load(content) or []

        # Append the new objects
        data.extend(objects)

        # Write the updated data back to the file asynchronously
        async with aiofiles.open(file_path, 'w') as file:
            await file.write(yaml.safe_dump(data, width=float('inf'), allow_unicode=True))

    except Exception as e:
        print(f"An error occurred: {e}")
