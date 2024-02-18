""" This module contains functions for generating data structures. """

from generation.structures.tree import BinaryTree, BinarySearchTree
from generation.structures.graph import UndirectedGraph, DirectedGraph
from utils.serializer import add_object
from utils.colors import Colors
from utils.logger import Logger
import os

from pathlib import Path
import yaml
from time import perf_counter
from typing import Type, Optional, TypeVar, Literal
import itertools
from uuid import uuid4, UUID

logger = Logger(__name__)

class Generator:
    
    Color = Literal['#88d7fe', '#feaf88', '#eeeeee']
    Shape = Literal['o', 's', 'd']
    Font = Literal['sans-serif', 'serif', 'monospace']
    Thickness = Literal['0.5', '1.0', '1.5']
    
    Structure = TypeVar('Structure', BinaryTree, BinarySearchTree, UndirectedGraph, DirectedGraph)

    def generate_structure(self, structure_class: Type[Structure], large: bool = False) -> Type[Structure]:
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

        print(Colors.OKBLUE + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" + Colors.ENDC)
        
        formal_name = structure_class.formal_name

        logger.info(f"Generating {formal_name}...")
        start = perf_counter()
        
        structure_instance = structure_class(large=large)
        structure_instance.generate()
        
        end = perf_counter()
        logger.info(f"╰── Generated {formal_name} in {round(end - start, 2)} seconds.")
        
        return structure_instance

    def fill_structure(self, structure_instance: Type[Structure]) -> Type[Structure]:
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
        print(Colors.OKBLUE + "┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄" + Colors.ENDC)
        
        formal_name = structure_instance.formal_name
        
        logger.info(f"Filling {formal_name}...")
        start = perf_counter()
        
        structure_instance.fill()
        
        end = perf_counter()
        logger.info(f"╰── Filled {formal_name} in {round(end - start, 2)} seconds.")
        
        return structure_instance

    def draw_structure(self, structure_instance: Type[Structure], uuid: UUID = None, text: str = None, expected: str = None, yaml: bool = False, yaml_path: Path = Path('.'), yaml_name: Optional[str] = None, save: bool = False, save_path: Path = Path('.'), save_name: Optional[str] = None, show: bool = True, run: int = 0, generation: int = 0, variation: int = 0, format: int = 0, shape: Shape = 'o', color: Color = '#88d7fe', font: Font = 'sans-serif', thickness: Thickness = '1.0') -> None:
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
        save_name : Optional[str] (default: None)
            the name of the image file
        show : bool (default: True)
            whether or not to show the image
        run : int (default: 0)
            the run number
        generation : int (default: 0)
            the generation number
        variation : int (default: 0)
            the variation number
        format : int (default: 0)
            the format number
        shape : Shape (default: 'o')
            the shape of the nodes
        color : Color (default: '#88d7fe')
            the color of the nodes
        font : Font (default: 'sans-serif')
            the font of the text
        thickness : Thickness (default: '1.0')
            the thickness of the edges
        
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            if generation, variation, or format is negative
        """
        print(Colors.OKBLUE + "┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄" + Colors.ENDC)
        
        default_file_name = structure_instance.default_file_name
        yaml_structure_type = structure_instance.yaml_structure_type
        formal_name = structure_instance.formal_name
        
        # Path handling
        save_path = Path(save_path)
        if not save_name.endswith('.png'):
            logger.warn("The save_name parameter does not end with '.png', so it will be added.")
            save_name += '.png'
        if not yaml_name.endswith('.yaml'):
            logger.warn("The yaml_name parameter does not end with '.yaml', so it will be added.")
            yaml_name += '.yaml'
        # Check for save flag
        if not save:
            if save_path != Path('.') or save_name != None:
                logger.warning("The save_path and save_name parameters are ignored since save is False.")
            if yaml:
                logger.warning("The yaml parameter is ignored since save is False.")
        # Check for yaml flag
        if not yaml:
            if yaml_path != Path('.') or yaml_name != None:
                logger.warning("The yaml_path and yaml_name parameters are ignored since yaml is False.")
            
        if save_name is None:
            save_name = default_file_name
            
        filepath = save_path / save_name
        
        # Parameter validation
        if run < 0 or generation < 0 or variation < 0 or format < 0:
            raise ValueError("Generation, variation, and format must be non-negative integers.")

        # Warning Checks
        if not save and not show:
            logger.warning("Neither save nor show is True, so the image will not be generated.")
            return
        
        if not save_path.exists():
            logger.error(f"Failed to save {formal_name} image because {save_path} does not exist.")
            return
        
        logger.info(f"Drawing {formal_name}...")
        start = perf_counter()
        
        structure_instance.draw(save=save, path=filepath, show=show, shape=shape, color=color, font=font, thickness=thickness)
        
        end = perf_counter()
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
                    
                    add_object(
                        uuid=uuid,
                        file_path=yaml_path,
                        text=text,
                        image_path=filepath,
                        expected=expected,
                        structure=yaml_structure_type,
                        run=run,
                        generation=generation,
                        variation=variation,
                        format=format,
                        color=color,
                        font=font,
                        thickness=thickness,
                    )
                    
                    logger.info(f"{formal_name} added to YAML file at {yaml_path}.")
                    
                except Exception as e:
                    logger.error(f"Failed to add binary tree to YAML: {e}")
                    
class BatchGenerator(Generator):
    
    # format attribute options
    colors = ['#88d7fe', '#feaf88', '#eeeeee']
    shapes = ['o', 's', 'd']
    fonts = ['sans-serif', 'serif', 'monospace']
    thickness = ['0.5', '1.0', '1.5']
    
    Structure = TypeVar('Structure', BinaryTree, BinarySearchTree, UndirectedGraph, DirectedGraph)
    
    StructureAbbreviation = Literal['bit', 'bst', 'udg', 'dig']
    YamlName = Literal['binary_tree.yaml', 'binary_search_tree.yaml', 'undirected_graph.yaml', 'directed_graph.yaml']
    
    generator: Generator
    
    def __init__(self) -> None:
        self.generator = Generator()

    def generate_batch(self, structure_class: Type[Structure], type: StructureAbbreviation, yaml_name: YamlName, yaml_path: Path, save_path: Path, text_path: Path, text_name: Path) -> None:
        
        run = 1
        
        for file_path in save_path.iterdir():
            if file_path.is_file() or file_path.is_symlink():
                file_path.unlink()
        
        text_path_joined = Path.joinpath(text_path, text_name)
        yaml_path_joined = Path.joinpath(yaml_path, yaml_name)
        
        with open(yaml_path_joined, 'w') as file:
            file.write('')
        
        with open(text_path_joined, 'r') as file:
            text_prompts = yaml.safe_load(file)
        
        # loop to create 5 base structures
        for generation in range(1, 2):
        
            structure_generated = self.generator.generate_structure(
                structure_class=structure_class,
                large=False,
            )
            
            # loop to create 3 variations of each base structure
            for variation in range(1, 2):
                
                structure_filled = self.generator.fill_structure(
                    structure_instance=structure_generated,
                )
                
                format = 1
                
                # loop to create 3 shape formats of each variation
                
                format_combinations = list(itertools.product(self.thickness, self.colors, self.fonts))
                
                for text in text_prompts:
                    
                    expected = None
                    
                    method_name = text['type']
                    if hasattr(structure_filled, method_name):
                        method_to_call = getattr(structure_filled, method_name)
                        expected = method_to_call()
                    else:
                        # Handle the case where the method does not exist
                        print(f"Method '{method_name}' not found in {structure_filled}.")
                
                    for thickness, color, font in format_combinations:
                        
                        uuid = uuid4()
                        
                        self.generator.draw_structure(
                            uuid=uuid,
                            structure_instance=structure_filled,
                            text=text['text'],
                            expected=str(expected),
                            yaml=True,
                            yaml_path=yaml_path,
                            yaml_name=yaml_name,
                            save=True,
                            save_path=save_path,
                            save_name=f"{type}_run-{run}_gen-{generation}_var-{variation}_fmt-{format}_thk-{thickness.replace('.', '')}_clr-{color.replace('#', '')}_fnt-{font.replace('-', '')}_idn-{str(uuid)}.png",
                            show=False,
                            run=run,
                            generation=generation,
                            variation=variation,
                            format=format,
                            color=color,
                            font=font,
                            thickness=thickness,
                        )
                        
                        format += 1
                        run += 1