from generators.structures.tree import BinaryTree, BinarySearchTree
from generators.structures.graph import UndirectedGraph, DirectedGraph

# TODO: Encode images as base64 strings and add them to the YAML file

def generate_binary_tree(large: bool = False, save: bool = False, path: str = '', filename: str = 'bt_test.png', show: bool = False):
    """
    Generate a binary tree.
    """
    
    filepath = path + filename
    
    binary_tree = BinaryTree(large=large)
    binary_tree_root = binary_tree.generate()
    binary_tree.draw(root=binary_tree_root, save=save, path=filepath, show=show)
    
    

def generate_binary_search_tree(large: bool = False, save: bool = False, path: str = '', filename: str = 'bst_test.png', show: bool = False):
    """
    Generate a binary search tree.
    """
    
    filepath = path + filename
    
    binary_search_tree = BinarySearchTree(large=large)
    binary_search_tree_root = binary_search_tree.generate()
    binary_search_tree.draw(root=binary_search_tree_root, save=save, path=filepath, show=show)
    
def generate_undirected_graph(large: bool = False, save: bool = False, path: str = '', filename: str = 'ug_test.png', show: bool = False):
    pass

def generate_directed_graph(large: bool = False, save: bool = False, path: str = '', filename: str = 'dg_test.png', show: bool = False):
    pass