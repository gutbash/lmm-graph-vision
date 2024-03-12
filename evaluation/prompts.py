# Aim: Realistic prompts by Students

# --------------------------------------------------------------------------------------------------------- #

# 1) Default
# Provide a single-line python list representing the post-order traversal of the {{structure}}.

# 2) Rephrase
# Attached is an image of a {{structure}} data structure. Provide its post-order traversal represented as a single-line python list.

# 3) No Structure
# Provide a single-line python list representing the post-order traversal.

# 4) Steps
# Provide a single-line python list representing the post-order traversal of the {{structure}} by following the steps below:                      
#    Algorithm Postorder(tree)
#    1. Traverse the left subtree, i.e., call Postorder(left->subtree)
#    2. Traverse the right subtree, i.e., call Postorder(right->subtree)
#    3. Visit the root

# 5) Definition
# Post-order traversal is a method of visiting all the nodes in a tree data structure where each node is visited after its subtrees have been visited.
# In post-order traversal, the nodes are visited in a left-right-root order. This traversal method is commonly used for operations to process a node 
#   only after having processed its subtrees, such as when calculating the size or height of the tree, or when deleting or freeing nodes from the tree.                             
# Provide a single-line python list representing the post-order traversal of the {{structure}}.

# 6) Expert
# You are a leading expert in data structures and algorithms, with a comprehensive understanding of both fundamental and advanced concepts.
#   Your expertise spans across various data structures such as arrays, linked lists, trees, graphs, stacks, queues, and hash tables,
#   as well as algorithms for searching, sorting, graph processing, and dynamic programming. You are adept at analyzing the time and
#   space complexity of algorithms and can effortlessly apply this knowledge to solve complex computational problems.
#   Your advice is sought after for optimizing code for efficiency, scalability, and performance. With your deep insights into theoretical
#   computer science, you can guide on the selection of the most appropriate data structures and algorithms for any given problem. You are
#   also familiar with real-world applications of these concepts in software development, data analysis, and system design.
# Provide a single-line python list representing the post-order traversal of the {{structure}}.

# 7) Reread
# Provide a single-line python list representing the post-order traversal of the {{structure}}.
#   Think step by step and read the prompt again carefully.
# Provide a single-line python list representing the post-order traversal of the {{structure}}.

# 8) Serial
# Input: {{structure}} Data Structure
# Input Type: Image File
# Output: Post-Order Traversal
# Output Type: Single-Line Python List

# 9) Polite
# Please provide a single-line python list representing the post-order traversal of the {{structure}}.

# 10) Zero-shot Chain of Thought: Encourage LLM to simulate human thought process
'''
Let's think step by step to understand the {{structure}} data structure image presented. 
How would you approach finding its **TASK** traversal? 
Your task is to provide a single-line Python list representing the **TASK** traversal of the {{structure}}.
'''

# 11) Golden Chain of Thought: embedding "ground-truth CoT" to predefine steps
'''
Given the visual representation of a {{structure}}, let's follow the step-by-step process below to **TASK**
Step 1: Identify the left subtree and perform a **TASK**.
Step 2: Move to the right subtree and complete its **TASK**.
Step 3: Visit the root of the {{structure}} last.
Following these steps closely, your task is to provide a single-line Python list that represents the post-order
    traversal of the {{structure}}, incorporating the sequence as if solving a puzzle step by step.
'''

# 12) Generate Knowledge: get LLM to generate information beforehand
'''
Before solving for the **TASK** of the given {{structure}},
    let's first generate some foundational knowledge that could aid in understanding the task better.
Consider the characteristics and properties of {{structure}} data structures.
What are the key aspects that influence their **traversal** order?
Utilize this generated knowledge to then provide a single-line Python list representing the **TASK** of the {{structure}}.
'''

# 13) Role Play: expert + CoT
'''
As an expert software engineer specializing in data structures and algorithms,
your mission is to analyze a visual representation of a {{structure}}.
Given your extensive experience, how would you explain and perform
a **TASK** of this structure? Your task is to provide a
single-line Python list representing the **TASK** of the {{structure}}.
'''

# 14) Delimiters: clear structure (###, ||, ), —, -), (||Task||[Desc], --Title--[Desc])
'''
###Instruction### [Given a {{structure}}, perform a post-order traversal.]
###Context### [The {{structure}} is a tree-like data structure visually represented in the attached image.]
###Task### [Provide a single-line Python list for the post-order traversal.]
'''

# 15) Golden Chain of Thought + Expert + Deliminator + Very Structured
'''
## Context ##
[ You are an expert computer scientist specializing in data structures and algorithms. You are given an image of a {{structure}} data structure to analyze. ]

## Task ##
[ Your task is to:
1. perform a **TASK** of the {{structure}}
2. provide a single-line Python list representing the **TRAVERSAL/SEARCH/DICTIONARY** sequence. 
Complete this task by following the step-by-step instructions closely.]

## Steps ##
[ Step 1: Identify the left subtree and perform a **TASK**.
Step 2: Move to the right subtree and complete its **TASK**.
Step 3: Visit the root of the {{structure}} last.
Step 4: Compile the results into a single-line Python list. ]
'''

# Any way we can incorporate number of nodes into the prompt?
# Unapplicable: Tree of Thoughts, Least-to-Most, Triple Quotes

# --------------------------------------------------------------------------------------------------------- #

from evaluation.models.messages.message import UserMessage, SystemMessage, ModelMessage

# 1) Default
PROMPTS_DEFAULT = [
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the post-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'post_order'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the pre-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the in-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'in_order'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python dictionary representing the adjacency list of the {{structure}}.""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the depth-first search of the {{structure}} starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the breadth-first search of the {{structure}} starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 2) Rephrase
PROMPTS_REPHRASE = [
    {
    'messages': [UserMessage(content="""Attached is an image of a {{structure}} data structure. Provide its post-order traversal represented as a single-line python list.""", images=["{{image}}"])],
    'task': 'post_order'
    },
    {
    'messages': [UserMessage(content="""Attached is an image of a {{structure}} data structure. Provide its pre-order traversal represented as a single-line python list.""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    {
    'messages': [UserMessage(content="""Attached is an image of a {{structure}} data structure. Provide its in-order traversal represented as a single-line python list.""", images=["{{image}}"])],
    'task': 'in_order'
    },
    {
    'messages': [UserMessage(content="""Attached is an image of a {{structure}} data structure. Provide its adjacency list represented as a single-line python dictionary.""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    {
    'messages': [UserMessage(content="""Attached is an image of a {{structure}} data structure. Provide its depth-first search represented as a single-line python list starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    {
    'messages': [UserMessage(content="""Attached is an image of a {{structure}} data structure. Provide its breadth-first search represented as a single-line python list starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 3) No Structure X
PROMPTS_NO_STRUCTURE = [
    {
    'messages': [UserMessage(content="Provide a single-line python list representing the post-order traversal.", images=["{{image}}"])],
    'task': 'post_order'
    },
    {
    'messages': [UserMessage(content="Provide a single-line python list representing the pre-order traversal.", images=["{{image}}"])],
    'task': 'pre_order'
    },
    {
    'messages': [UserMessage(content="Provide a single-line python list representing the in-order traversal.", images=["{{image}}"])],
    'task': 'in_order'
    },
    {
    'messages': [UserMessage(content="Provide a single-line python dictionary representing the adjacency list.", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    {
    'messages': [UserMessage(content="Provide a single-line python list representing the depth-first search starting from the vertex with the smallest value.", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    {
    'messages': [UserMessage(content="Provide a single-line python list representing the breadth-first search starting from the vertex with the smallest value.", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 4) Steps
PROMPTS_STEPS = [
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the post-order traversal of the {{structure}} by following the steps below:
                             
    Algorithm Postorder(tree)
    
    1. Traverse the left subtree, i.e., call Postorder(left->subtree)
    2. Traverse the right subtree, i.e., call Postorder(right->subtree)
    3. Visit the root""", images=["{{image}}"])],
    'task': 'post_order'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the pre-order traversal of the {{structure}} by following the steps below:
                             
    Algorithm Preorder(tree)

    1. Visit the root.
    2. Traverse the left subtree, i.e., call Preorder(left->subtree)
    3. Traverse the right subtree, i.e., call Preorder(right->subtree)""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the in-order traversal of the {{structure}} by following the steps below:
                             
    Algorithm Inorder(tree)
    
    1. Traverse the left subtree, i.e., call Inorder(left->subtree)
    2. Visit the root.
    3. Traverse the right subtree, i.e., call Inorder(right->subtree)""", images=["{{image}}"])],
    'task': 'in_order'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python dictionary representing the adjacency list of the {{structure}} by following the steps below:
    
    1. Initialize a dictionary to represent the graph, with each vertex as a key and its adjacent vertices as a list for the corresponding value.
    2. For every vertex in the graph, assign an empty list to the dictionary entry for that vertex to hold its adjacent vertices.
    3. Iterate through all the edges in the graph, represented as pairs (u, v). For each pair, append v to the list of u's adjacent vertices in the dictionary. If the graph is undirected, also append u to v's list.""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the depth-first search of the {{structure}} starting from the vertex with the smallest value.
    
    Algorithm DFS(graph, startVertex)

    1. Initialize a stack S and push startVertex onto S.
    2. While S is not empty:
        a. Pop a vertex v from S.
        b. If v has not been visited:
            i. Mark v as visited.
            ii. For each neighbor w of v in graph:
                - If w has not been visited, push w onto S.""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the breadth-first search of the {{structure}} starting from the vertex with the smallest value.
    
    Algorithm BFS(graph, startVertex)

    1. Create a queue Q and enqueue startVertex into Q.
    2. Mark startVertex as visited.
    3. While Q is not empty:
        a. Dequeue a vertex v from Q.
        b. Visit v.
        c. For each neighbor w of v in Graph:
            i. If w is not visited:
                - Mark w as visited.
                - Enqueue w into Q.""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 5) Definition
PROMPTS_DEFINITION = [
    {
    'messages': [UserMessage(content="""Post-order traversal is a method of visiting all the nodes in a tree data structure where each node is visited after its subtrees have been visited.
                             
    In post-order traversal, the nodes are visited in a left-right-root order. This traversal method is commonly used for operations to process a node only after having processed its subtrees, such as when calculating the size or height of the tree, or when deleting or freeing nodes from the tree.
                             
    Provide a single-line python list representing the post-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'post_order'
    },
    {
    'messages': [UserMessage(content="""Pre-order traversal is a technique used to visit all the nodes in a tree data structure where each node is visited before its subtrees.
                             
    In pre-order traversal, the nodes are visited in a root-left-right order. This method is used in operations that need to copy or examine the structure of the tree itself, such as in the cloning of trees or in the serialization of trees where a node needs to be processed before its descendants.
                             
    Provide a single-line python list representing the pre-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    {
    'messages': [UserMessage(content="""In-order traversal is a method of visiting all the nodes in a binary tree based on a left-root-right sequence.
                             
    The in-order traversal method ensures that nodes are visited in non-decreasing order of their key values in a binary search tree, making it particularly useful for operations that require processing tree nodes in their logical order, such as printing all elements in a binary search tree in sorted order.
                             
    Provide a single-line python list representing the in-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'in_order'
    },
    {
    'messages': [UserMessage(content="""An adjacency list is a collection of lists used to represent a graph. Each list corresponds to a vertex in the graph and contains a list of all the vertices that are adjacent to it. This method of graph representation is efficient in terms of space, especially for sparse graphs where the number of edges is much less than the square of the number of vertices. It allows for quick lookup to find all the neighbors of any vertex.
                             
    Provide a single-line python dictionary representing the adjacency list of the {{structure}}.""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    {
    'messages': [UserMessage(content="""Depth-first search is an algorithm for traversing or searching tree or graph data structures. The algorithm starts at the root node (selecting some arbitrary node as the root in the case of a graph) and explores as far as possible along each branch before backtracking. The basic idea is to start from the root (or any arbitrary node) and mark the node and move to an adjacent unmarked node and continue this loop until there is no unmarked adjacent node. Then backtrack and check for other unmarked nodes and traverse them. Finally, print the nodes in the path.

    DFS can be implemented using recursion and backtracking or with a stack and is used in algorithms that need to explore all the nodes, such as checking for cycle in graphs, path finding, and solving puzzles with only one solution, like mazes.
                             
    Provide a single-line python list representing the depth-first search of the {{structure}} starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    {
    'messages': [UserMessage(content="""Breadth-first search is an algorithm for traversing or searching tree or graph data structures. It starts at an arbitrary node of a graph and explores the neighbor nodes first, before moving to the next level neighbors. BFS uses a queue to keep track of the nodes that are to be explored.
                             
    BFS is used for finding the shortest path on unweighted graphs, as it guarantees the minimum number of edges that must be traversed to reach a node from the starting point.
                             
    Provide a single-line python list representing the breadth-first search of the {{structure}} starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 6) Expert
PROMPTS_EXPERT = [
    {
    'messages': [UserMessage(content="""You are a leading expert in data structures and algorithms, with a comprehensive understanding of both fundamental and advanced concepts. Your expertise spans across various data structures such as arrays, linked lists, trees, graphs, stacks, queues, and hash tables, as well as algorithms for searching, sorting, graph processing, and dynamic programming. You are adept at analyzing the time and space complexity of algorithms and can effortlessly apply this knowledge to solve complex computational problems. Your advice is sought after for optimizing code for efficiency, scalability, and performance. With your deep insights into theoretical computer science, you can guide on the selection of the most appropriate data structures and algorithms for any given problem. You are also familiar with real-world applications of these concepts in software development, data analysis, and system design.
                             
    Provide a single-line python list representing the post-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'post_order'
    },
    {
    'messages': [UserMessage(content="""You are a leading expert in data structures and algorithms, with a comprehensive understanding of both fundamental and advanced concepts. Your expertise spans across various data structures such as arrays, linked lists, trees, graphs, stacks, queues, and hash tables, as well as algorithms for searching, sorting, graph processing, and dynamic programming. You are adept at analyzing the time and space complexity of algorithms and can effortlessly apply this knowledge to solve complex computational problems. Your advice is sought after for optimizing code for efficiency, scalability, and performance. With your deep insights into theoretical computer science, you can guide on the selection of the most appropriate data structures and algorithms for any given problem. You are also familiar with real-world applications of these concepts in software development, data analysis, and system design.
                             
    Provide a single-line python list representing the pre-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    {
    'messages': [UserMessage(content="""You are a leading expert in data structures and algorithms, with a comprehensive understanding of both fundamental and advanced concepts. Your expertise spans across various data structures such as arrays, linked lists, trees, graphs, stacks, queues, and hash tables, as well as algorithms for searching, sorting, graph processing, and dynamic programming. You are adept at analyzing the time and space complexity of algorithms and can effortlessly apply this knowledge to solve complex computational problems. Your advice is sought after for optimizing code for efficiency, scalability, and performance. With your deep insights into theoretical computer science, you can guide on the selection of the most appropriate data structures and algorithms for any given problem. You are also familiar with real-world applications of these concepts in software development, data analysis, and system design.
                             
    Provide a single-line python list representing the in-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'in_order'
    },
    {
    'messages': [UserMessage(content="""You are a leading expert in data structures and algorithms, with a comprehensive understanding of both fundamental and advanced concepts. Your expertise spans across various data structures such as arrays, linked lists, trees, graphs, stacks, queues, and hash tables, as well as algorithms for searching, sorting, graph processing, and dynamic programming. You are adept at analyzing the time and space complexity of algorithms and can effortlessly apply this knowledge to solve complex computational problems. Your advice is sought after for optimizing code for efficiency, scalability, and performance. With your deep insights into theoretical computer science, you can guide on the selection of the most appropriate data structures and algorithms for any given problem. You are also familiar with real-world applications of these concepts in software development, data analysis, and system design.
                             
    Provide a single-line python dictionary representing the adjacency list of the {{structure}}.""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    {
    'messages': [UserMessage(content="""You are a leading expert in data structures and algorithms, with a comprehensive understanding of both fundamental and advanced concepts. Your expertise spans across various data structures such as arrays, linked lists, trees, graphs, stacks, queues, and hash tables, as well as algorithms for searching, sorting, graph processing, and dynamic programming. You are adept at analyzing the time and space complexity of algorithms and can effortlessly apply this knowledge to solve complex computational problems. Your advice is sought after for optimizing code for efficiency, scalability, and performance. With your deep insights into theoretical computer science, you can guide on the selection of the most appropriate data structures and algorithms for any given problem. You are also familiar with real-world applications of these concepts in software development, data analysis, and system design.
                             
    Provide a single-line python list representing the depth-first search of the {{structure}} starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    {
    'messages': [UserMessage(content="""You are a leading expert in data structures and algorithms, with a comprehensive understanding of both fundamental and advanced concepts. Your expertise spans across various data structures such as arrays, linked lists, trees, graphs, stacks, queues, and hash tables, as well as algorithms for searching, sorting, graph processing, and dynamic programming. You are adept at analyzing the time and space complexity of algorithms and can effortlessly apply this knowledge to solve complex computational problems. Your advice is sought after for optimizing code for efficiency, scalability, and performance. With your deep insights into theoretical computer science, you can guide on the selection of the most appropriate data structures and algorithms for any given problem. You are also familiar with real-world applications of these concepts in software development, data analysis, and system design.
                             
    Provide a single-line python list representing the breadth-first search of the {{structure}} starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 7) Reread + Zero Shot CoT
PROMPTS_ZERO_SHOT_COT_REREAD = [
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the post-order traversal of the {{structure}}.
                             
    Read the prompt again: Provide a single-line python list representing the post-order traversal of the {{structure}}.
    
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'post_order'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the pre-order traversal of the {{structure}}.
                             
    Read the prompt again: Provide a single-line python list representing the pre-order traversal of the {{structure}}.
    
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the in-order traversal of the {{structure}}.
                             
    Read the prompt again: Provide a single-line python list representing the in-order traversal of the {{structure}}.
    
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'in_order'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python dictionary representing the adjacency list of the {{structure}}.
                             
    Read the prompt again: Provide a single-line python dictionary representing the adjacency list of the {{structure}}.
    
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the depth-first search of the {{structure}} starting from the vertex with the smallest value.
                             
    Read the prompt again: Provide a single-line python list representing the depth-first search of the {{structure}} starting from the vertex with the smallest value.
    
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the breadth-first search of the {{structure}} starting from the vertex with the smallest value.
                             
    Read the prompt again: Provide a single-line python list representing the breadth-first search of the {{structure}} starting from the vertex with the smallest value.
    
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 8) Serial
PROMPTS_SERIAL = [
    {
    'messages': [UserMessage(content="""Input: {{structure}} Data Structure
    Input Type: Image File
    Output: Post-Order Traversal
    Output Type: Single-Line Python List""", images=["{{image}}"])],
    'task': 'post_order'
    },
    {
    'messages': [UserMessage(content="""Input: {{structure}} Data Structure
    Input Type: Image File
    Output: Pre-Order Traversal
    Output Type: Single-Line Python List""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    {
    'messages': [UserMessage(content="""Input: {{structure}} Data Structure
    Input Type: Image File
    Output: In-Order Traversal
    Output Type: Single-Line Python List""", images=["{{image}}"])],
    'task': 'in_order'
    },
    {
    'messages': [UserMessage(content="""Input: {{structure}} Data Structure
    Input Type: Image File
    Output: Adjacency List
    Output Type: Single-Line Python Dictionary""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    {
    'messages': [UserMessage(content="""Input: {{structure}} Data Structure
    Input Type: Image File
    Output: Depth-First Search (start at vertex with smallest value)
    Output Type: Single-Line Python List""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    {
    'messages': [UserMessage(content="""Input: {{structure}} Data Structure
    Input Type: Image File
    Output: Breadth-First Search (start at vertex with smallest value)
    Output Type: Single-Line Python List""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 9) Polite X
PROMPTS_POLITE = [
    {
    'messages': [UserMessage(content="""Please provide a single-line python list representing the post-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'post_order'
    },
    {
    'messages': [UserMessage(content="""Please provide a single-line python list representing the pre-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    {
    'messages': [UserMessage(content="""Please provide a single-line python list representing the in-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'in_order'
    },
    {
    'messages': [UserMessage(content="""Please provide a single-line python dictionary representing the adjacency list of the {{structure}}.""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    {
    'messages': [UserMessage(content="""Please provide a single-line python list representing the depth-first search of the {{structure}} starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    {
    'messages': [UserMessage(content="""Please provide a single-line python list representing the breadth-first search of the {{structure}} starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 10) Zero-shot Chain-of-Thought
PROMPTS_ZERO_SHOT_COT = [
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the post-order traversal of the {{structure}}.
                             
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'post_order'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the pre-order traversal of the {{structure}}.
                             
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the in-order traversal of the {{structure}}.
                             
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'in_order'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python dictionary representing the adjacency list of the {{structure}}.
                             
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the depth-first search of the {{structure}} starting from the vertex with the smallest value.
                             
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    {
    'messages': [UserMessage(content="""Provide a single-line python list representing the breadth-first search of the {{structure}} starting from the vertex with the smallest value.
                             
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 11) Golden Chain-of-Thought
PROMPTS_GOLD_COT = [
    { # post-order
    'messages': [UserMessage(content="""Given the image of a {{structure}}, let's follow the step-by-step process below to perform a post-order traversal.
                             
    Step 1: Identify the left subtree and perform a post-order traversal.
    Step 2: Move to the right subtree and complete its post-order traversal.
    Step 3: Visit the root of the {{structure}} last.
    
    Following these steps closely, your task is to provide a single-line Python list that represents the post-order traversal of the {{structure}}, incorporating the sequence as if solving a puzzle step by step.""", images=["{{image}}"])],
    'task': 'post_order'
    },
    { # pre-order
    'messages': [UserMessage(content="""Given the image of a {{structure}}, let's follow the step-by-step process below to perform a pre-order traversal.
                             
    Step 1: Visit the root of the {{structure}} first.
    Step 2: Move to the left subtree and perform a pre-order traversal.
    Step 3: Proceed to the right subtree and complete its pre-order traversal.
    
    Following these steps closely, your task is to provide a single-line Python list that represents the pre-order traversal of the {{structure}}, incorporating the sequence as if solving a puzzle step by step.""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    { # in-order
    'messages': [UserMessage(content="""Given the image of a {{structure}}, let's follow the step-by-step process below to perform an in-order traversal.
                             
    Step 1: Start with the left subtree and perform an in-order traversal.
    Step 2: Visit the root of the {{structure}}.
    Step 3: Proceed to the right subtree and complete its in-order traversal.
    
    Following these steps closely, your task is to provide a single-line Python list that represents the in-order traversal of the {{structure}}, incorporating the sequence as if solving a puzzle step by step.""", images=["{{image}}"])],
        'task': 'in_order'
    },
    { # Adjacency List
    'messages': [UserMessage(content="""Given the image of a {{structure}}, let's follow the step-by-step process below to create an adjacency list.
                             
    Step 1: Initialize a dictionary to represent the graph, with each vertex as a key and its adjacent vertices as a list for the corresponding value.
    Step 2: For every vertex in the graph, assign an empty list to the dictionary entry for that vertex to hold its adjacent vertices.
    Step 3: Iterate through all the edges in the graph, represented as pairs (u, v). For each pair, append v to the list of u's adjacent vertices in the dictionary. If the graph is undirected, also append u to v's list.
    
    Following these steps closely, your task is to provide a single-line Python dictionary that represents the adjacency list of the {{structure}}, incorporating the sequence as if solving a puzzle step by step.""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    { # depth-first Search
    'messages': [UserMessage(content="""Given the image of a {{structure}}, let's follow the step-by-step process below to perform a depth-first search.
                             
    Step 1: Initialize a stack S and push the start vertex onto S.
    Step 2: While S is not empty, pop a vertex v from S. If v has not been visited, mark it as visited and push all of its unvisited neighbors onto S.
    Step 3: Repeat Step 2 until the stack is empty or the desired node is found.
    
    Following these steps closely, your task is to provide a single-line Python list that represents the depth-first search of the {{structure}} starting from the vertex with the smallest value, incorporating the sequence as if solving a puzzle step by step.""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    { # breadth-first Search
    'messages': [UserMessage(content="""Given the image of a {{structure}}, let's follow the step-by-step process below to perform a breadth-first search.
                             
    Step 1: Create a queue Q and enqueue the start vertex into Q. Mark the start vertex as visited.
    Step 2: While Q is not empty, dequeue a vertex v from Q. Visit v and enqueue all unvisited neighbors of v into Q.
    Step 3: Repeat Step 2 until the queue is empty or the desired node is found.
    
    Following these steps closely, your task is to provide a single-line Python list that represents the breadth-first search of the {{structure}} starting from the vertex with the smallest value, incorporating the sequence as if solving a puzzle step by step.""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 12) Generate Knowledge X
PROMPTS_GENERAL_KNOWLEDGE = [
    { # post-order
    'messages': [UserMessage(content="""Before solving for the post-order traversal of the given {{structure}}, let's first generate some foundational knowledge that could aid in understanding the task better. Consider the characteristics and properties of {{structure}} data structures. What are the key aspects that influence their traversal order? Utilize this generated knowledge to then provide a single-line Python list representing the post-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'post_order'
    },
    { # pre-order
    'messages': [UserMessage(content="""Before solving for the pre-order traversal of the given {{structure}}, let's first generate some foundational knowledge that could aid in understanding the task better. Consider the characteristics and properties of {{structure}} data structures. What are the key aspects that influence their traversal order? Utilize this generated knowledge to then provide a single-line Python list representing the pre-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    { # in-order
    'messages': [UserMessage(content="""Before solving for the in-order traversal of the given {{structure}}, let's first generate some foundational knowledge that could aid in understanding the task better. Consider the characteristics and properties of {{structure}} data structures. What are the key aspects that influence their traversal order? Utilize this generated knowledge to then provide a single-line Python list representing the in-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'in_order'
    },
    { # Adjacency List
    'messages': [UserMessage(content="""Before solving for the adjacency list of the given {{structure}}, let's first generate some foundational knowledge that could aid in understanding the task better. Consider the characteristics and properties of {{structure}} data structures. What are the key aspects that influence their adjacency list? Utilize this generated knowledge to then provide a single-line Python dictionary representing the adjacency list of the {{structure}}.""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    { # depth-first Search
    'messages': [UserMessage(content="""Before solving for the depth-first search of the given {{structure}}, let's first generate some foundational knowledge that could aid in understanding the task better. Consider the characteristics and properties of {{structure}} data structures. What are the key aspects that influence their search order? Utilize this generated knowledge to then provide a single-line Python list representing the depth-first search of the {{structure}} starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    { # breadth-first Search
    'messages': [UserMessage(content="""Before solving for the breadth-first search of the given {{structure}}, let's first generate some foundational knowledge that could aid in understanding the task better. Consider the characteristics and properties of {{structure}} data structures. What are the key aspects that influence their search order? Utilize this generated knowledge to then provide a single-line Python list representing the breadth-first search of the {{structure}} starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 13) Role Play: expert + CoT X
PROMPTS_ROLEPLAY_EXPERT_COT = [
    { # post-order
    'messages': [UserMessage(content="""As an expert software engineer specializing in data structures and algorithms, your mission is to analyze a image of a {{structure}}. Given your extensive experience, how would you explain and perform a post-order traversal of this structure? Your task is to provide a single-line Python list representing the post-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'post_order'
    },
    { # pre-order
    'messages': [UserMessage(content="""As an expert software engineer specializing in data structures and algorithms, your mission is to analyze a image of a {{structure}}. Given your extensive experience, how would you explain and perform a pre-order traversal of this structure? Your task is to provide a single-line Python list representing the pre-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    { # in-order
    'messages': [UserMessage(content="""As an expert software engineer specializing in data structures and algorithms, your mission is to analyze a image of a {{structure}}. Given your extensive experience, how would you explain and perform a in-order traversal of this structure? Your task is to provide a single-line Python list representing the in-order traversal of the {{structure}}.""", images=["{{image}}"])],
    'task': 'in_order'
    },
    { # Adjacency List
    'messages': [UserMessage(content="""As an expert software engineer specializing in data structures and algorithms, your mission is to analyze a image of a {{structure}}. Given your extensive experience, how would you explain and create an adjacency list of this structure? Your task is to provide a single-line Python list representing the dictionary of the {{structure}}.""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    { # depth-first Search
    'messages': [UserMessage(content="""As an expert software engineer specializing in data structures and algorithms, your mission is to analyze a image of a {{structure}}. Given your extensive experience, how would you explain and perform a depth-first search of this structure? Your task is to provide a single-line Python list representing the depth-first search of the {{structure}} starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    { # breadth-first Search
    'messages': [UserMessage(content="""As an expert software engineer specializing in data structures and algorithms, your mission is to analyze a image of a {{structure}}. Given your extensive experience, how would you explain and perform a breadth-first search of this structure? Your task is to provide a single-line Python list representing the breadth-first search of the {{structure}} starting from the vertex with the smallest value.""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 14) Delimiters: clear structure X
PROMPTS_DELIMIT = [
    { # post-order
    'messages': [UserMessage(content="""###Instruction### [Given a {{structure}}, perform a post-order traversal.]
    ###Context### [The {{structure}} is the data structure that is visually represented in the attached image.]
    ###Task### [Provide a single-line Python list for the post-order traversal.]""", images=["{{image}}"])],
    'task': 'post_order'
    },
    { # pre-order
    'messages': [UserMessage(content="""###Instruction### [Given a {{structure}}, perform a pre-order traversal.]
    ###Context### [The {{structure}} is the data structure that is visually represented in the attached image.]
    ###Task### [Provide a single-line Python list for the pre-order traversal.]""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    { # in-order
    'messages': [UserMessage(content="""###Instruction### [Given a {{structure}}, perform a in-order traversal.]
    ###Context### [The {{structure}} is the data structure that is visually represented in the attached image.]
    ###Task### [Provide a single-line Python list for the in-order traversal.]""", images=["{{image}}"])],
        'task': 'in_order'
    },
    { # Adjacency List
    'messages': [UserMessage(content="""###Instruction### [Given a {{structure}}, create a adjacency list.]
    ###Context### [The {{structure}} is the data structure that is visually represented in the attached image.]
    ###Task### [Provide a single-line Python dictionary for the adjacency list.]""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    { # depth-first Search
    'messages': [UserMessage(content="""###Instruction### [Given a {{structure}}, perform a depth-first search.]
    ###Context### [The {{structure}} is the data structure that is visually represented in the attached image.]
    ###Task### [Provide a single-line Python list for the depth-first search starting from the vertex with the smallest value.]""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    { # breadth-first Search
    'messages': [UserMessage(content="""###Instruction### [Given a {{structure}}, perform a breadth-first search.]
    ###Context### [The {{structure}} is the data structure that is visually represented in the attached image.]
    ###Task### [Provide a single-line Python list for the breadth-first search starting from the vertex with the smallest value.]""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

# 15) Golden Chain of Thought + Expert + Deliminator + Very Structured
PROMPTS_GOLD_COT_EXPERT_DELIMIT = [
    { # post-order
    'messages': [UserMessage(content="""## Context ##
    You are an expert computer scientist specializing in data structures and algorithms. You are given an image of a {{structure}} data structure to analyze.

    ## Task ##
    Your task is to...
    
    1. Perform a post-order traversal of the {{structure}}.
    2. Provide a single-line Python list representing the traversal.
    
    Complete this task by following the step-by-step instructions closely.

    ## Steps ##
    Step 1: Identify the left subtree and perform a post-order traversal.
    Step 2: Move to the right subtree and complete its post-order traversal.
    Step 3: Visit the root of the {{structure}} last.
    Step 4: Compile the results into a single-line Python list.""", images=["{{image}}"])],
    'task': 'post_order'
    },
    { # pre-order
    'messages': [UserMessage(content="""## Context ##
    You are an expert computer scientist specializing in data structures and algorithms. You are given an image of a {{structure}} data structure to analyze.

    ## Task ##
    Your task is to...
    
    1. Perform a pre-order traversal of the {{structure}}.
    2. Provide a single-line Python list representing the traversal.
    
    Complete this task by following the step-by-step instructions closely.

    ## Steps ##
    Step 1: Visit the root of the {{structure}} first.
    Step 2: Move to the left subtree and perform a pre-order traversal.
    Step 3: Proceed to the right subtree and complete its pre-order traversal.
    Step 4: Compile the results into a single-line Python list.""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    { # in-order
    'messages': [UserMessage(content="""## Context ##
    You are an expert computer scientist specializing in data structures and algorithms. You are given an image of a {{structure}} data structure to analyze.

    ## Task ##
    Your task is to...
    
    1. Perform a in-order traversal of the {{structure}}.
    2. Provide a single-line Python list representing the traversal.
    
    Complete this task by following the step-by-step instructions closely.

    ## Steps ##
    Step 1: Start with the left subtree and perform an in-order traversal.
    Step 2: Visit the root of the {{structure}}.
    Step 3: Proceed to the right subtree and complete its in-order traversal.
    Step 4: Compile the results into a single-line Python list.""", images=["{{image}}"])],
    'task': 'in_order'
    },
    { # Adjacency List
    'messages': [UserMessage(content="""## Context ##
    You are an expert computer scientist specializing in data structures and algorithms. You are given an image of a {{structure}} data structure to analyze.

    ## Task ##
    Your task is to...
    
    1. Create an adjacency list of the {{structure}}.
    2. Provide a single-line Python list representing the dictionary.
    
    Complete this task by following the step-by-step instructions closely.

    ## Steps ##
    Step 1: Initialize a dictionary to represent the graph, with each vertex as a key and its adjacent vertices as a list for the corresponding value.
    Step 2: For every vertex in the graph, assign an empty list to the dictionary entry for that vertex to hold its adjacent vertices.
    Step 3: Iterate through all the edges in the graph, represented as pairs (u, v). For each pair, append v to the list of u's adjacent vertices in the dictionary. If the graph is undirected, also append u to v's list.
    Step 4: Compile the results into a single-line Python list.""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    { # depth-first Search
    'messages': [UserMessage(content="""## Context ##
    You are an expert computer scientist specializing in data structures and algorithms. You are given an image of a {{structure}} data structure to analyze.

    ## Task ##
    Your task is to...
    
    1. Perform a depth-first search of the {{structure}} starting from the vertex with the smallest value.
    2. Provide a single-line Python list representing the search.
    
    Complete this task by following the step-by-step instructions closely.

    ## Steps ##
    Step 1: Initialize a stack S and push the start vertex onto S.
    Step 2: While S is not empty, pop a vertex v from S. If v has not been visited, mark it as visited and push all of its unvisited neighbors onto S.
    Step 3: Repeat Step 2 until the stack is empty or the desired node is found.
    Step 4: Compile the results into a single-line Python list.""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    { # breadth-first Search
    'messages': [UserMessage(content="""## Context ##
    You are an expert computer scientist specializing in data structures and algorithms. You are given an image of a {{structure}} data structure to analyze.

    ## Task ##
    Your task is to...
    
    1. Perform a breadth-first search of the {{structure}} starting from the vertex with the smallest value.
    2. Provide a single-line Python list representing the search.
    
    Complete this task by following the step-by-step instructions closely.

    ## Steps ##
    Step 1: Initialize a stack S and push the start vertex onto S.
    Step 2: While S is not empty, pop a vertex v from S. If v has not been visited, mark it as visited and push all of its unvisited neighbors onto S.
    Step 3: Repeat Step 2 until the stack is empty or the desired node is found.
    Step 4: Compile the results into a single-line Python list.""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

PROMPTS_ZERO_SHOT_COT_POLITE = [
    {
    'messages': [UserMessage(content="""Please provide a single-line python list representing the post-order traversal of the {{structure}}.
                             
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'post_order'
    },
    {
    'messages': [UserMessage(content="""Please provide a single-line python list representing the pre-order traversal of the {{structure}}.
                             
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'pre_order'
    },
    {
    'messages': [UserMessage(content="""Please provide a single-line python list representing the in-order traversal of the {{structure}}.
                             
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'in_order'
    },
    {
    'messages': [UserMessage(content="""Please provide a single-line python dictionary representing the adjacency list of the {{structure}}.
                             
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    {
    'messages': [UserMessage(content="""Please provide a single-line python list representing the depth-first search of the {{structure}} starting from the vertex with the smallest value.
                             
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    {
    'messages': [UserMessage(content="""Please provide a single-line python list representing the breadth-first search of the {{structure}} starting from the vertex with the smallest value.
                             
    Let's think step by step.""", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]
