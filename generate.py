from generators.tree import BinaryTree, BinarySearchTree

# Instantiate a binary tree
binary_tree = BinaryTree(large=True)
# Generate a binary tree 
binary_tree_root = binary_tree.generate()
# Draw the binary tree
binary_tree.draw(binary_tree_root)

# Instantiate a binary search tree
binary_search_tree = BinarySearchTree()
# Generate a binary search tree
binary_search_tree_root = binary_search_tree.generate()
# Draw the binary search tree
#binary_search_tree.draw(binary_search_tree_root)