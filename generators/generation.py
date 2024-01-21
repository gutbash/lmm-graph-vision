""" This module contains functions for generating data structures. """

from generators.structures.tree import BinaryTree, BinarySearchTree
from generators.structures.graph import UndirectedGraph, DirectedGraph
from images.encoding import encode_image
from data.operations import add_object

# TODO: Encode images as base64 strings and add them to the YAML file

def generate_binary_tree(large: bool = False, yaml: bool = False, save: bool = False, path: str = '', filename: str = 'bt_test.png', show: bool = False, generation: int = 0, variation: int = 0, format: int = 0):
    """
    Generate a binary tree.
    
    Parameters
    ----------
    large : bool (default: False)
        whether or not the binary tree is large
    yaml : bool (default: False)
        whether or not to add the object to the YAML file
    save : bool (default: False)
        whether or not to save the image
    path : str (default: '')
        the path to the image
    filename : str (default: 'bt_test.png')
        the name of the image file
    show : bool (default: False)
        whether or not to show the image
    generation : int (default: 1)
        the generation number
    variation : int (default: 1)
        the variation number
    format : int (default: 1)
        the format number
    
    Returns
    -------
    None
    
    Raises
    ------
    None
    """
    
    filepath = path + filename
    
    binary_tree = BinaryTree(large=large)
    binary_tree_root = binary_tree.generate()
    binary_tree.draw(root=binary_tree_root, save=save, path=filepath, show=show)
    
    if yaml:
        encoded_image = encode_image(image_path=filepath)
        
        add_object(
            file_path='data/tree/binary_tree.yaml',
            text=None,
            image_data=encoded_image,
            image_path=filepath,
            expected=None,
            structure='binary_tree',
            generation=generation,
            variation=variation,
            format=format
        )
    

def generate_binary_search_tree(large: bool = False, yaml: bool = False, save: bool = False, path: str = '', filename: str = 'bst_test.png', show: bool = False, generation: int = 0, variation: int = 0, format: int = 0):
    """
    Generate a binary search tree.
    
    Parameters
    ----------
    large : bool (default: False)
        whether or not the binary search tree is large
    yaml : bool (default: False)
        whether or not to add the object to the YAML file
    save : bool (default: False)
        whether or not to save the image
    path : str (default: '')
        the path to the image
    filename : str (default: 'bst_test.png')
        the name of the image file
    show : bool (default: False)
        whether or not to show the image
    generation : int (default: 1)
        the generation number
    variation : int (default: 1)
        the variation number
    format : int (default: 1)
        the format number
    
    Returns
    -------
    None
    
    Raises
    ------
    None
    """
    
    filepath = path + filename
    
    binary_search_tree = BinarySearchTree(large=large)
    binary_search_tree_root = binary_search_tree.generate()
    binary_search_tree.draw(root=binary_search_tree_root, save=save, path=filepath, show=show)
    
    if yaml:
        encoded_image = encode_image(image_path=filepath)
        
        add_object(
            file_path='data/tree/binary_search_tree.yaml',
            text=None,
            image_data=encoded_image,
            image_path=filepath,
            expected=None,
            structure='binary_search_tree',
            generation=generation,
            variation=variation,
            format=format
        )
    
def generate_undirected_graph(large: bool = False, save: bool = False, path: str = '', filename: str = 'ug_test.png', show: bool = False):
    pass

def generate_directed_graph(large: bool = False, save: bool = False, path: str = '', filename: str = 'dg_test.png', show: bool = False):
    pass