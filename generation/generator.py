"""Contains classes for generating data structures."""

from generation.structures.tree import BinaryTree, BinarySearchTree
from generation.structures.graph import UndirectedGraph, DirectedGraph
from utils.serializer import add_objects_async
from utils.colors import Colors
from utils.logger import Logger
from utils.files import validate_path, check_path_exists, has_negative_value
import os

from pathlib import Path
import yaml
from time import perf_counter
from typing import Type, Optional, TypeVar, Literal
import itertools
from uuid import uuid4, UUID
import inspect

logger = Logger(__name__)

Color = Literal['#abe0f9', '#fee4b3', '#eeeeee']
Shape = Literal['o', 's', 'd']
Font = Literal['sans-serif', 'serif', 'monospace']
Width = Literal['0.5', '1.0', '1.5']

Structure = TypeVar('Structure', BinaryTree, BinarySearchTree, UndirectedGraph, DirectedGraph)

StructureAbbreviation = Literal['bit', 'bst', 'udg', 'dig']
YamlName = Literal['binary_tree.yaml', 'binary_search_tree.yaml', 'undirected_graph.yaml', 'directed_graph.yaml']

class Generator:
    """
    Generator class for generating data structures.

    Example
    -------
    ```python
    from generation.generator import Generator
    generator = Generator()
    ```
    """

    async def generate_structure(self, structure_class: Type[Structure], num_nodes: int = None, num_edges: int = None) -> Type[Structure]:
        """
        Generates an empty structure instance and returns it.
        
        Parameters
        ----------
        structure_class : Type[Structure]
            the structure to generate
            
        Returns
        -------
        Type[Structure]
            the generated structure instance
        """

        print(Colors.OKBLUE + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" + Colors.ENDC)

        logger.info(f"Generating {structure_class.formal_name}...")
        start = perf_counter()
        
        structure_instance = structure_class()
        # Check if 'generate' method accepts 'num_edges'
        if 'num_edges' in inspect.signature(structure_instance.generate).parameters:
            structure_instance.generate(num_nodes=num_nodes, num_edges=num_edges)
        else:
            structure_instance.generate(num_nodes=num_nodes)

        end = perf_counter()
        logger.info(f"╰── Generated {structure_class.formal_name} in {round(end - start, 2)} seconds.")
        
        return structure_instance

    async def fill_structure(self, structure_instance: Type[Structure]) -> Type[Structure]:
        """
        Fills the structure instance and returns it.
        
        Parameters
        ----------
        structure_instance : Type[Structure]
            the structure instance to fill
            
        Returns
        -------
        Type[Structure]
            the filled structure instance
        """
        print(Colors.OKBLUE + "┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄" + Colors.ENDC)
        
        logger.info(f"Filling {structure_instance.formal_name}...")
        start = perf_counter()
        
        structure_instance.fill() # fill structure
        
        end = perf_counter()
        logger.info(f"╰── Filled {structure_instance.formal_name} in {round(end - start, 2)} seconds.")
        
        return structure_instance

    async def draw_structure(self, structure_instance: Type[Structure], image_id: UUID = None, task_id: UUID = None, task_type: str = None, expected: str = None, save: bool = False, save_path: Path = Path('.'), save_name: Optional[str] = None, show: bool = True, generation: int = 0, variation: int = 0, format: int = 0, shape: Shape = 'o', color: Color = '#fee4b3', font: Font = 'sans-serif', width: Width = '1.5', num_nodes: int = 0, resolution: int = 512, arrow_style: str = '-|>') -> None:
        """
        Draws the structure instance and saves the image to a file and/or adds the object to a YAML file.
        
        Parameters
        ----------
        structure_instance : Type[Structure]
            the structure instance to fill
        task_id : UUID (default: None)
            the UUID of the object
        expected : str (default: None)
            the expected output of the model
        save : bool (default: False)
            whether or not to save the image
        save_path : Path (default: Path('.'))
            the path to the image
        save_name : Optional[str] (default: None)
            the name of the image file
        show : bool (default: True)
            whether or not to show the image
        generation : int (default: 0)
            the generation number
        variation : int (default: 0)
            the variation number
        format : int (default: 0)
            the format number
        shape : Shape (default: 'o')
            the shape of the nodes
        color : Color (default: '#abe0f9')
            the color of the nodes
        font : Font (default: 'sans-serif')
            the font of the text
        width : Width (default: '1.0')
            the width of the edges
        
        Raises
        ------
        ValueError
            if generation, variation, or format is negative
        """
        print(Colors.OKBLUE + "┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄" + Colors.ENDC)

        if not save:
            if not show:
                logger.error("Neither save nor show is True, so the image will not be generated.")
                return
            if save_path != Path('.') or save_name:
                logger.warning("The save_path and save_name parameters are ignored since save is False.")

        if not save_name:
            save_name = structure_instance.default_file_name
            
        filepath = validate_path(save_path, save_name, '.png')

        if has_negative_value([variation, format]):
            raise ValueError("Generation, variation, and format must be non-negative integers.")
        
        logger.info(f"Drawing {structure_instance.formal_name}...")
        start = perf_counter()
        
        structure_instance.draw(save=save, path=filepath, show=show, shape=shape, color=color, font=font, width=width, resolution=resolution, arrow_style=arrow_style)
        
        end = perf_counter()
        logger.info(f"╰── Drew {structure_instance.formal_name} in {round(end - start, 2)} seconds.")
        
        if save:
            logger.info(f"{structure_instance.formal_name} image saved to {filepath}.")
        return {
            'image_id': str(image_id),
            'task_id': str(task_id),
            'task': task_type,
            'ground_truth': expected,
            'structure': structure_instance.yaml_structure_type,
            'generation_id': generation,
            'variation_id': variation,
            'format_id': format,
            'image_path': str(filepath),
            'node_color': color,
            'node_font': font,
            'node_shape': shape,
            'edge_width': float(width),
            'arrow_style': arrow_style,
            'num_nodes': num_nodes,
            'resolution': resolution,
        }
                    
class BatchGenerator(Generator):
    """
    Generate data structures in batches.
    
    Example
    -------
    ```python
    from generation.generator import BatchGenerator
    batch_generator = BatchGenerator()
    ```
    """
    generator: Generator
    
    def __init__(self) -> None:
        """
        Initializes the BatchGenerator class.
        """
        self.generator = Generator()

    async def generate_batch(self, structure_class: Type[Structure], type: StructureAbbreviation, yaml_name: YamlName, yaml_path: Path, save_path: Path, generations: int = 1, variations: int = 1, random_num_nodes: bool = False, num_edges: Optional[list] = None, resolutions: list = [512], arrows: list = ['-|>'], colors: list = ['#fee4b3'], shapes: list = ['o'], fonts: list = ['sans-serif'], width: list = ['1.5']) -> None:
        """
        Generates a batch of data structures.
        
        Parameters
        ----------
        structure_class : Type[Structure]
            the structure to generate
        type : StructureAbbreviation
            the abbreviation of the structure
        yaml_name : YamlName
            the name of the YAML file
        yaml_path : Path
            the path to the YAML file
        save_path : Path
            the path to save the images
        """
        yaml_objects = []
        format = 1
        
        check_path_exists(save_path)
        
        for file_path in save_path.iterdir():
            if file_path.is_file() or file_path.is_symlink():
                file_path.unlink()
        
        yaml_path_joined = validate_path(yaml_path, yaml_name, '.yaml')
        
        with open(yaml_path_joined, 'w') as file:
            file.write('')
            
        format_combinations = list(itertools.product(width, colors, fonts, shapes))
        
        # create base structures
        for generation in range(1, generations + 1):
            
            for subgeneration in range(3):
            
                generation_approved = False
                structure_generated = None
                num_nodes = generation + 2 if not random_num_nodes else None
                
                while not generation_approved:
                    
                    test_path = check_path_exists(Path('images/'))
                    test_name = 'test.png'

                    # get the number of edges for the current generation
                    current_num_edges = num_edges[generation - 1] if num_edges and generation <= len(num_edges) else None

                    structure_generated = await self.generator.generate_structure(structure_class=structure_class, num_nodes=num_nodes, num_edges=current_num_edges)
                    structure_filled = await self.generator.fill_structure(structure_instance=structure_generated)
                    await self.generator.draw_structure(
                        structure_instance=structure_filled,
                        save=True,
                        save_path=test_path,
                        save_name=test_name,
                        show=False,
                    )
                    
                    logger.warning(f"[G{generation}-{subgeneration}N{num_nodes}] Check {test_path / test_name}. Approve this generation?\n\n(Y) Approved, continue generating\n(N) Denied, regenerate\n(X) Exit\n")
                    input_approved = input(">>> ")
                    
                    if input_approved.lower() == 'y':
                        generation_approved = True
                    elif input_approved.lower() == 'x':
                        return

                # create variations of each base structure
                for variation in range(1, variations + 1):

                    format = 1

                    structure_filled = await self.generator.fill_structure(structure_instance=structure_generated)
                    
                    for width, color, font, shape in format_combinations:
                        
                        for arw, arrow in enumerate(arrows):

                            for res in resolutions:
                                
                                image_id = uuid4()
                        
                                for method in structure_filled.methods:

                                    task_id = uuid4()
                                    
                                    expected = None
                                    if hasattr(structure_filled, method):
                                        method_to_call = getattr(structure_filled, method)
                                        expected = method_to_call(structure_filled)
                                    else:
                                        logger.error(f"Method '{method}' not found in {structure_filled}.")
                                
                                    object = await self.generator.draw_structure(
                                        image_id=image_id,
                                        task_id=task_id,
                                        structure_instance=structure_filled,
                                        task_type=method,
                                        expected=str(expected),
                                        save=True,
                                        save_path=save_path,
                                        save_name=f"{type}-gen_{generation}_{subgeneration}-var_{variation}-fmt_{format}-wid_{width.replace('.', '')}-col_{color.replace('#', '')}-fnt_{font.replace('-', '')}-shp_{shape}-arw_{arw}-res_{str(res)}-idn_{str(image_id)}.png",
                                        show=False,
                                        generation=f"{generation}-{subgeneration}",
                                        variation=variation,
                                        format=format,
                                        color=color,
                                        font=font,
                                        width=width,
                                        shape=shape,
                                        num_nodes=len(structure_filled.graph.nodes),
                                        resolution=res,
                                        arrow_style=arrow,
                                    )
                                
                                    yaml_objects.append(object)
                            
                        format += 1
                        
        try:
            await add_objects_async(file_path=yaml_path_joined, objects=yaml_objects)
            logger.info(f"{structure_filled.formal_name} added to YAML file at {yaml_path_joined}.")
        except Exception as e:
            logger.error(f"Failed to add binary tree to YAML: {e}")