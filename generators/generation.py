""" This module contains functions for generating data structures. """

from images.encoding import encode_image
from data.operations import add_object

from pathlib import Path
import time
from typing import Type, Any
from colorlog import ColoredFormatter
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Create a console handler
ch = logging.StreamHandler()

# Create a formatter with color
formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

# Add formatter to console handler
ch.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(ch)

def generate_structure(structure: Type[Any], large: bool = False, yaml: bool = False, yaml_path: Path = Path('.'), yaml_name: str = None, save: bool = False, save_path: Path = Path('.'), file_name: str = None, show: bool = True, generation: int = 0, variation: int = 0, format: int = 0) -> None:
    """
    Generate a binary tree.
    
    Parameters
    ----------
    structure : Type[Any]
        the structure to generate (e.g. BinaryTree, BinarySearchTree, UndirectedGraph, DirectedGraph)
    large : bool (default: False)
        whether or not the binary tree is large
    yaml : bool (default: False)
        whether or not to add the object to a YAML file
    yaml_path : Path (default: Path('.'))
        the path to the YAML file
    yaml_name : str (default: None)
        the name of the YAML file
    save : bool (default: False)
        whether or not to save the image
    save_path : Path (default: Path('.'))
        the path to the image
    file_name : str (default: None)
        the name of the image file
    show : bool (default: True)
        whether or not to show the image
    generation : int (default: 0)
        the generation number
    variation : int (default: 0)
        the variation number
    format : int (default: 0)
        the format number
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        if structure is not a valid structure (e.g. BinaryTree, BinarySearchTree, UndirectedGraph, DirectedGraph)
        if generation, variation, or format is negative
    """
    
    print()
    
    # Path handling
    save_path = Path(save_path)
    if not file_name.endswith('.png'):
        logger.warn("The file_name parameter does not end with '.png', so it will be added.")
        file_name += '.png'
    if not yaml_name.endswith('.yaml'):
        logger.warn("The yaml_name parameter does not end with '.yaml', so it will be added.")
        yaml_name += '.yaml'
    # Check for save flag
    if not save:
        if save_path != Path('.') or file_name != None:
            logger.warning("The save_path and file_name parameters are ignored since save is False.")
        if yaml:
            logger.warning("The yaml parameter is ignored since save is False.")
    # Check for yaml flag
    if not yaml:
        if yaml_path != Path('.') or yaml_name != None:
            logger.warning("The yaml_path and yaml_name parameters are ignored since yaml is False.")
    
    structure_name = structure.__name__

    match structure_name:
        case 'BinaryTree':
            default_file_name = 'bt_test.png'
            yaml_structure_type = 'binary_tree'
            formal_name = 'Binary Tree'
        case 'BinarySearchTree':
            default_file_name = 'bst_test.png'
            yaml_structure_type = 'binary_search_tree'
            formal_name = 'Binary Search Tree'
        case 'UndirectedGraph':
            default_file_name = 'ug_test.png'
            yaml_structure_type = 'undirected_graph'
            formal_name = 'Undirected Graph'
        case 'DirectedGraph':
            default_file_name = 'dg_test.png'
            yaml_structure_type = 'directed_graph'
            formal_name = 'Directed Graph'
        case _:
            raise ValueError("Structure must be a valid structure (e.g. BinaryTree, BinarySearchTree, UndirectedGraph, DirectedGraph).")
        
    if file_name is None:
        file_name = default_file_name
        
    filepath = save_path / file_name
    
    # Parameter validation
    if generation < 0 or variation < 0 or format < 0:
        raise ValueError("Generation, variation, and format must be non-negative integers.")

    # Warning Checks
    if not save and not show:
        logger.warning("Neither save nor show is True, so the image will not be generated.")
        return
    
    logger.info(f"Generating {formal_name}...")
    start = time.time()
    
    instantiated_structure = structure(large=large)
    instantiated_structure.generate()
    instantiated_structure.draw(save=save, path=filepath, show=show)
    
    end = time.time()
    logger.info(f"{formal_name} generated in {round(end - start, 2)} seconds.")
    
    if save:
        if not save_path.exists():
            logger.error(f"Failed to save {formal_name} image because {save_path} does not exist.")
            return
        else:
            logger.info(f"{formal_name} image saved to {filepath}.")
        if yaml:
            
            if not yaml_path.exists():
                logger.error(f"Failed to add {formal_name} to YAML because {yaml_path} does not exist.")
                return
            
            if yaml_name is None:
                yaml_name = 'test.yaml'
                
            yaml_path = yaml_path / yaml_name
            
            try:
                encoded_image = encode_image(image_path=filepath)
                
                add_object(
                    file_path=yaml_path,
                    text=None,
                    image_data=encoded_image,
                    image_path=filepath,
                    expected=None,
                    structure=yaml_structure_type,
                    generation=generation,
                    variation=variation,
                    format=format
                )
                
                logger.info(f"{formal_name} added to YAML file at {yaml_path}.")
                
            except Exception as e:
                logger.error(f"Failed to add binary tree to YAML: {e}")