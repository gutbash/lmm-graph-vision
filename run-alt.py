import os
from pathlib import Path
from dotenv import load_dotenv

from generation.generator import Generator, BatchGenerator
from generation.structures.tree import BinaryTree, BinarySearchTree
from generation.structures.graph import UndirectedGraph, DirectedGraph

from evaluation.evaluator import Evaluator
from evaluation.models.openai import OpenAI
from evaluation.models.deepmind import DeepMind
from evaluation.models.messages.message import UserMessage, SystemMessage, AssistantMessage, ImageMessage, BaseMessage

class EnvironmentSetup:
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY_DEV')
        self.deepmind_api_key = os.getenv('DEEPMIND_API_KEY_DEV')
        self.setup_paths()

    def setup_paths(self):
        self.image_path = {
            'binary_tree': Path('images/binary_tree/'),
            'binary_search_tree': Path('images/binary_search_tree/'),
            'undirected_graph': Path('images/undirected_graph/'),
            'directed_graph': Path('images/directed_graph/')
        }
        self.yaml_path = Path('data/')
        self.text_path = Path('text/')
        self.results_path = Path('results/')
        self.create_directories()

    def create_directories(self):
        for path in self.image_path.values():
            path.mkdir(parents=True, exist_ok=True)
        self.yaml_path.mkdir(parents=True, exist_ok=True)
        self.text_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)

class BatchGeneration:
    def __init__(self, paths):
        self.paths = paths
        self.batch_generator = BatchGenerator()
        
        self.class_to_path = { # internal mapping from structure classes to path keys
            BinaryTree: 'binary_tree',
            BinarySearchTree: 'binary_search_tree',
            UndirectedGraph: 'undirected_graph',
            DirectedGraph: 'directed_graph',
        }

    def generate_batches(self, selected_structure_classes, generations=1, variations=1):
        for structure_class in selected_structure_classes:
            path_key = self.class_to_path[structure_class] # get path key
            self.batch_generator.generate_batch(
                structure_class=structure_class,
                type=structure_class.__name__.lower()[:2],
                yaml_name=f'{path_key}.yaml',
                yaml_path=self.paths.yaml_path,
                save_path=self.paths.image_path[path_key],
                text_path=self.paths.text_path,
                text_name=f'{path_key}_text.yaml',
                generations=generations,
                variations=variations,
            )

class Evaluation:
    def __init__(self, api_keys, paths):
        self.models = {
            'openai': OpenAI(api_key=api_keys['openai']),
            'deepmind': DeepMind(api_key=api_keys['deepmind'])
        }
        self.paths = paths

    def evaluate_models(self, selected_structure_names):
        evaluator = Evaluator()
        for model_name, structure_name in selected_structure_names:
            model = self.models[model_name]

            if model_name == 'openai':
                messages = [
                    SystemMessage(content="You are GPT-4V. A large multimodal model with vision capabilities trained by OpenAI."),
                    UserMessage(content="{{content}}", images=["{{image}}"]),
                ]

            elif model_name == 'deepmind':
                messages = [
                    BaseMessage(content="{{content}}"),
                    ImageMessage(image="{{image}}"),
                ]

            else:
                continue
            
            yaml_name = f'{structure_name}.yaml'
            csv_name = f'{model_name}_{structure_name}.csv'
            evaluator.evaluate(
                model = model,
                messages = messages,
                #limit = 3, # <-- unsure about this, some were limit=none
                yaml_path = self.paths.yaml_path,
                yaml_name = yaml_name,
                csv_path = self.paths.results_path,
                csv_name = csv_name
            )

def pre_order_check(node): # Root, Left, Right
    if node is None:
        return []
    return [node.value] + pre_order_check(node.left) + pre_order_check(node.right)

def in_order_check(node): # Left, Root, Right
    if node is None:
        return []
    return in_order_check(node.left) + [node.value] + in_order_check(node.right)

def post_order_check(node): # Left, Right, Root
    if node is None:
        return []
    return post_order_check(node.left) + post_order_check(node.right) + [node.value]

def test_BTree_traversal():
    # PreOrder = Root, Left, Right
    # InORder = Left, Root, Right
    # PostOrder = Left, Right, Root
    binary_tree = BinaryTree()
    print("==={ Tree Traversal Test }===")
    binary_tree.generate()
    print("Generation Complete")
    binary_tree.fill()
    print("Fill Complete")
    binary_tree.draw()
    print("Draw Complete")

    # Tree Traversals - [!] Tree has Randomized Vals from Fill -> can't compare...
    pre_order_result = binary_tree.pre_order(binary_tree)
    print("[TREE] Pre-order traversal:", ' -> '.join(map(str, pre_order_result)))
    in_order_result = binary_tree.in_order(binary_tree)
    print("[TREE] In-order traversal:", ' -> '.join(map(str, in_order_result)))
    post_order_result = binary_tree.post_order(binary_tree)
    print("[TREE] Post-order traversal:", ' -> '.join(map(str, post_order_result)))

    # Local Traversals 
    pre_order_check_result = pre_order_check(binary_tree.root)
    print("[LOCAL] Pre-order traversal:", ' -> '.join(map(str, pre_order_check_result)))
    in_order_check_result = in_order_check(binary_tree.root)
    print("[LOCAL] In-order traversal:", ' -> '.join(map(str, in_order_check_result)))
    post_order_check_result = post_order_check(binary_tree.root)
    print("[LOCAL] Post-order traversal:", ' -> '.join(map(str, post_order_check_result)))

    # Compare Traversals
    if pre_order_result == pre_order_check_result:
        print("Pre-order traversal matches.")
    else:
        print("Pre-order traversal does not match.")

    if in_order_result == in_order_check_result:
        print("In-order traversal matches.")
    else:
        print("In-order traversal does not match.")

    if post_order_result == post_order_check_result:
        print("Post-order traversal matches.")
    else:
        print("Post-order traversal does not match.")

def test_BSTree_traversal():
    # PreOrder = Root, Left, Right
    # InORder = Left, Root, Right
    # PostOrder = Left, Right, Root
    bst = BinarySearchTree()
    print("==={ Tree Traversal Test }===")
    bst.generate()
    print("Generation Complete")
    bst.fill()
    print("Fill Complete")
    bst.draw()
    print("Draw Complete")

    # Tree Traversals
    pre_order_result = bst.pre_order(bst)   # KeyError: 44
                                            # Key doesn't exist in 'graph.nodes' dictionary?
    print("[TREE] Pre-order traversal:", ' -> '.join(map(str, pre_order_result)))
    in_order_result = bst.in_order(bst)
    print("[TREE] In-order traversal:", ' -> '.join(map(str, in_order_result)))
    post_order_result = bst.post_order(bst)
    print("[TREE] Post-order traversal:", ' -> '.join(map(str, post_order_result)))

def test_UndirectedGraph_traversals():
    print("==={ Undirected Graph Test }===")
    ug = UndirectedGraph()
    ug.generate(num_nodes=10)
    print("Generation Complete")
    ug.fill()
    print("Fill Complete")
    ug.draw()
    print("Draw Complete")

    # Graph Operations
    adjacency_list = ug.adjacency_list(ug)
    bfs_result = ug.breadth_first_search(ug)
    dfs_result = ug.depth_first_search(ug)

    print("[GRAPH] Adjacency List:", adjacency_list)
    print("[GRAPH] BFS traversal:", ' -> '.join(map(str, bfs_result)))
    print("[GRAPH] DFS traversal:", ' -> '.join(map(str, dfs_result)))

    # Local Traversals
    local_bfs_result = local_bfs(adjacency_list, 1)
    local_dfs_result = local_dfs(adjacency_list, 1)
    print("[LOCAL] BFS traversal using manual method:", ' -> '.join(map(str, local_bfs_result)))
    print("[LOCAL] DFS traversal using manual method:", ' -> '.join(map(str, local_dfs_result)))

    # Compare the results
    print("BFS same:", bfs_result == local_bfs_result)
    print("DFS same:", dfs_result == local_dfs_result)

def test_DirectedGraph_traversals():
    print("==={ Directed Graph Test }===")
    dg = DirectedGraph()
    dg.generate(num_nodes=10)
    print("Generation Complete")
    dg.fill()
    print("Fill Complete")
    dg.draw()
    print("Draw Complete")

    # Graph Operations
    adjacency_list = dg.adjacency_list(dg)
    bfs_result = dg.breadth_first_search(dg)
    dfs_result = dg.depth_first_search(dg)

    print("[GRAPH] Adjacency List:", adjacency_list)
    print("[GRAPH] BFS traversal:", ' -> '.join(map(str, bfs_result)))
    print("[GRAPH] DFS traversal:", ' -> '.join(map(str, dfs_result)))

    # Local Traversals
    local_bfs_result = local_bfs(adjacency_list, 1)
    local_dfs_result = local_dfs(adjacency_list, 1)
    print("[LOCAL] BFS traversal using manual method:", ' -> '.join(map(str, local_bfs_result)))
    print("[LOCAL] DFS traversal using manual method:", ' -> '.join(map(str, local_dfs_result)))

    # Compare the results
    print("BFS same:", bfs_result == local_bfs_result)
    print("DFS same:", dfs_result == local_dfs_result)

def local_bfs(graph, start):
    visited = []
    queue = [start]

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            neighbours = graph[node]
            for neighbour in neighbours:
                queue.append(neighbour)
    return visited

def local_dfs(graph, start, visited=None):
    if visited is None:
        visited = []
    visited.append(start)
    for neighbour in graph[start]:
        if neighbour not in visited:
            local_dfs(graph, neighbour, visited)
    return visited


if __name__ == "__main__":
    setup = EnvironmentSetup()

    # [GENERATE DATA STRUCTURES]
    # !! <Select Data Structure to Generate>
    selected_structures = [
        BinaryTree,
        # BinarySearchTree,
        # UndirectedGraph,
        # DirectedGraph,
    ]
    # !! <Select Num of Generations/Variations>
    generations = 1 # Generations = Nodes in Diff Places
    variations = 1 # Variations = Diff Numbers
    #batch_gen = BatchGeneration(setup)
    #batch_gen.generate_batches(selected_structures, generations, variations)

    # [EVALUATE DATA STRUCTURES]
    eval = Evaluation({'openai': setup.openai_api_key, 'deepmind': setup.deepmind_api_key}, setup)
    # !! <Select Data Structure to Evaluate>
    selected_structure_names = [
        # DeepMind evaluations
        #('deepmind', 'binary_tree'),
        #('deepmind', 'binary_search_tree'),
        #('deepmind', 'undirected_graph'),
        #('deepmind', 'directed_graph'),
        
        # OpenAI evaluations
        #('openai', 'binary_tree'),
        #('openai', 'binary_search_tree'),
        #('openai', 'undirected_graph'),
        #('openai', 'directed_graph'),
    ]
    #eval.evaluate_models(selected_structure_names)

    # [TEST TREE TRAVERSAL] #
    #test_BTree_traversal() # WORKING x10
    test_BSTree_traversal() # Traversals KeyError
    #test_UndirectedGraph_traversals() # WORKING x5; WORKING WITH LOCAL x5
    #test_DirectedGraph_traversals() # WORKING x5; WORKING WITH LOCAL x5