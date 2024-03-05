from evaluation.models.messages.message import UserMessage, SystemMessage, AssistantMessage, ImageMessage, BaseMessage

OPENAI_PROMPTS = [
    {
    'messages': [UserMessage(content="Provide a single-line python list representing the post-order traversal of the {{structure}}.", images=["{{image}}"])],
    'task': 'post_order'
    },
    {
    'messages': [UserMessage(content="Provide a single-line python list representing the pre-order traversal of the {{structure}}.", images=["{{image}}"])],
    'task': 'pre_order'
    },
    {
    'messages': [UserMessage(content="Provide a single-line python list representing the in-order traversal of the {{structure}}.", images=["{{image}}"])],
    'task': 'in_order'
    },
    {
    'messages': [UserMessage(content="Provide a single-line python dictionary representing the adjacency list of the {{structure}}.", images=["{{image}}"])],
    'task': 'adjacency_list'
    },
    {
    'messages': [UserMessage(content="Provide a single-line python list representing the depth-first search of the {{structure}} starting from the vertex with the smallest value.", images=["{{image}}"])],
    'task': 'depth_first_search'
    },
    {
    'messages': [UserMessage(content="Provide a single-line python list representing the breadth-first search of the {{structure}} starting from the vertex with the smallest value.", images=["{{image}}"])],
    'task': 'breadth_first_search'
    },
]

DEEPMIND_PROMPTS = [
    {
    'messages': [BaseMessage(content="Provide a single-line python list representing the post-order traversal of the {{structure}}."), ImageMessage(image="{{image}}")],
    'task': 'post_order'
    },
    {
    'messages': [BaseMessage(content="Provide a single-line python list representing the pre-order traversal of the {{structure}}."), ImageMessage(image="{{image}}")],
    'task': 'pre_order'
    },
    {
    'messages': [BaseMessage(content="Provide a single-line python list representing the in-order traversal of the {{structure}}."), ImageMessage(image="{{image}}")],
    'task': 'in_order'
    },
    {
    'messages': [BaseMessage(content="Provide a single-line python dictionary representing the adjacency list of the {{structure}}."), ImageMessage(image="{{image}}")],
    'task': 'adjacency_list'
    },
    {
    'messages': [BaseMessage(content="Provide a single-line python list representing the depth-first search of the {{structure}} starting from the vertex with the smallest value."), ImageMessage(image="{{image}}")],
    'task': 'depth_first_search'
    },
    {
    'messages': [BaseMessage(content="Provide a single-line python list representing the breadth-first search of the {{structure}} starting from the vertex with the smallest value."), ImageMessage(image="{{image}}")],
    'task': 'breadth_first_search'
    },
]