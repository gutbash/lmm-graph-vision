""" This module contains functions for generating data structures. """

from generators.structures.tree import BinaryTree, BinarySearchTree
from generators.structures.graph import UndirectedGraph, DirectedGraph
from images.encoding import encode_image
from data.operations import add_object

from pathlib import Path
import time
from typing import Type, Optional, TypeVar, Literal
from colorlog import ColoredFormatter
import logging

Color = Literal['#88d7fe', '#feaf88', '#eeeeee']
Shape = Literal['o', 's', 'd']
Font = Literal['sans-serif', 'serif', 'monospace']

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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

Structure = TypeVar('Structure', BinaryTree, BinarySearchTree, UndirectedGraph, DirectedGraph)

def generate_structure(structure_class: Type[Structure], large: bool = False) -> Type[Structure]:
    """
    Generates an empty structure instance and returns it
    
    Parameters
    ----------
    structure_class : Type[Structure]
        the structure to generate
    large : bool (default: False)
        whether or not the structure should be large
        
    Returns
    -------
    Type[Structure]
        the generated structure instance
        
    Raises
    ------
    None
    """

    print(bcolors.OKBLUE + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" + bcolors.ENDC)
    
    formal_name = structure_class.formal_name

    logger.info(f"Generating {formal_name}...")
    start = time.perf_counter()
    
    structure_instance = structure_class(large=large)
    structure_instance.generate()
    
    end = time.perf_counter()
    logger.info(f"╰── Generated {formal_name} in {round(end - start, 2)} seconds.")
    
    return structure_instance

def fill_structure(structure_instance: Type[Structure]) -> Type[Structure]:
    """
    Fills the structure instance and returns it
    
    Parameters
    ----------
    structure_instance : Type[Structure]
        the structure instance to fill
        
    Returns
    -------
    Type[Structure]
        the filled structure instance
        
    Raises
    ------
    None
    """
    print(bcolors.OKBLUE + "┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄" + bcolors.ENDC)
    
    formal_name = structure_instance.formal_name
    
    logger.info(f"Filling {formal_name}...")
    start = time.perf_counter()
    
    structure_instance.fill()
    
    end = time.perf_counter()
    logger.info(f"╰── Filled {formal_name} in {round(end - start, 2)} seconds.")
    
    return structure_instance

def draw_structure(structure_instance: Type[Structure], yaml: bool = False, yaml_path: Path = Path('.'), yaml_name: Optional[str] = None, save: bool = False, save_path: Path = Path('.'), file_name: Optional[str] = None, show: bool = True, generation: int = 0, variation: int = 0, format: int = 0, shape: Shape = 'o', color: Color = '#88d7fe', font: Font = 'sans-serif') -> None:
    """
    Draws the structure instance and saves the image to a file and/or adds the object to a YAML file
    
    Parameters
    ----------
    structure_instance : Type[Structure]
        the structure instance to fill
    yaml : bool (default: False)
        whether or not to add the object to a YAML file
    yaml_path : Path (default: Path('.'))
        the path to the YAML file
    yaml_name : Optional[str] (default: None)
        the name of the YAML file
    save : bool (default: False)
        whether or not to save the image
    save_path : Path (default: Path('.'))
        the path to the image
    file_name : Optional[str] (default: None)
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
        if generation, variation, or format is negative
    """
    print(bcolors.OKBLUE + "┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄" + bcolors.ENDC)
    
    default_file_name = structure_instance.default_file_name
    yaml_structure_type = structure_instance.yaml_structure_type
    formal_name = structure_instance.formal_name
    
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
    
    if not save_path.exists():
        logger.error(f"Failed to save {formal_name} image because {save_path} does not exist.")
        return
    
    logger.info(f"Drawing {formal_name}...")
    start = time.perf_counter()
    
    structure_instance.draw(save=save, path=filepath, show=show, shape=shape, color=color, font=font)
    
    end = time.perf_counter()
    logger.info(f"╰── Drew {formal_name} in {round(end - start, 2)} seconds.")
    
    if save:
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