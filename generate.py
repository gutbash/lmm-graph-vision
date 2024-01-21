from generators.tree import BinaryTree, BinarySearchTree
from generators.graph import UndirectedGraph, DirectedGraph

# Instantiate a binary tree
binary_tree = BinaryTree(large=True)
# Generate a binary tree 
binary_tree_root = binary_tree.generate()
# Draw the binary tree
binary_tree.draw(root=binary_tree_root, save=True, path='output.png', show=False)

# Instantiate a binary search tree
binary_search_tree = BinarySearchTree()
# Generate a binary search tree
binary_search_tree_root = binary_search_tree.generate()
# Draw the binary search tree
#binary_search_tree.draw(root=binary_search_tree_root, save=True, path='output.png', show=False)