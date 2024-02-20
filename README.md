# lmm-graph-tree-vqa

How well do the **GPT-4V** and **Gemini Pro Vision** models perform **Visual Question Answering (VQA)** on **Data Structures**?

There is no dataset for **VQA** on graph and tree data structures in previous work, so we must create one. We create a standard, repeatable process for selecting and obtaining **VQA** tasks that fall under a certain criteria.

## Setup

The following instructions use a **bash** terminal and assume you have [Python](https://www.python.org/downloads/) and [Git](https://git-scm.com/downloads) installed on your machine.

1. Clone the repository
    ```bash
    git clone https://github.com/gutbash/lmm-graph-tree-vqa.git
    cd lmm-graph-tree-vqa
    ```
2. Create a virtual environment
    ```bash
    python -m venv .venv
    ```
3. Activate the virtual environment

    Linux and macOS:
    ```bash
    source .venv/bin/activate
    ```
    Windows:
    ```bash
    source .venv/Scripts/activate
    ```
4. Install the dependencies
    ```bash
    pip install -r requirements.txt
    ```
5. Set Environment Variables
    ```bash
    mv .env.example .env
    ```
    Edit the `.env` file and set the environment variables.

## Quickstart

<details closed>
<summary>Structures</summary>

### Structures

>At the core of the project are the data structures. These are the base structures that are used to generate images for the **VQA** tasks.

There are four base classes: `BinaryTree`, `BinarySearchTree`, `DirectedGraph`, `UndirectedGraph`.

You can generate an individual image directly from these classes, but it is not the conventional approach.

The following example generates an image of a binary tree:
```python
from generation.structures.tree import BinaryTree

structure = BinaryTree()

structure.generate()
structure.fill()
structure.draw(save=True, path='test.png')
```
</details>

<details closed>
<summary>Generators</summary>

### Generators
>Generate an individual image.

The conventional way to generate an individual image is to use the `Generator`.

The following example does the same as the previous example:
```python
from generation.structures.tree import BinaryTree
from generation.generator import Generator
from pathlib import Path

generator = Generator()

generated = generator.generate_structure(structure_class=BinaryTree)
filled = generator.fill_structure(structure_instance=generated)
generator.draw_structure(structure_instance=filled, save=True, save_path=Path('test/'), save_name='test.png')
```
</details>

<details closed>
<summary>Batch Generators</summary>

### Batch Generators
>Generate a batch of images.

Use the `BatchGenerator` to create a batch of images. This will also link text and image prompts into the `yaml` data.

The following example generates a batch of binary trees:
```python
from generation.structures.tree import BinaryTree
from generation.generator import BatchGenerator
from pathlib import Path

batch_generator = BatchGenerator()

batch_generator.generate_batch(
    structure_class=BinaryTree,
    type='bit',
    yaml_name='binary_tree.yaml',
    yaml_path=Path('data/'),
    save_path=Path('images/binary_tree/'),
    text_path=Path('text/'),
    text_name='binary_tree_text.yaml',
)
```
</details>

<details closed>
<summary>Models</summary>

### Models

>Create instances of models for evaluation.

There are two models that can be created for evaluation: `OpenAI` and `DeepMind`.

The following example creates instances of both models:
```python
from evaluation.models.openai import OpenAI
from evaluation.models.deepmind import DeepMind
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY_DEV')
deepmind_api_key = os.environ.get('DEEPMIND_API_KEY_DEV')

openai = OpenAI(api_key=openai_api_key)
deepmind = DeepMind(api_key=deepmind_api_key)
```
</details>

<details closed>
<summary>Messages</summary>

### Messages

>Build a template for a prompt for a model with a list of messages.

**OpenAI** can use the following message types for prompts: `SystemMessage`, `UserMessage`, and `AssistantMessage`.

The following example creates a typical prompt for **OpenAI**:
```python
from evaluation.models.messages.message import UserMessage, SystemMessage, AssistantMessage
from pathlib import Path

openai_messages = [
    UserMessage(content="Answer this question: What is in this image?", images=[Path('test/test.png')]),
]
```

**DeepMind** can use the following message types for prompts: `ImageMessage` and `BaseMessage`.

The following example creates a typical prompt for **DeepMind**:
```python
from evaluation.models.messages.message import ImageMessage, BaseMessage
from pathlib import Path

deepmind_messages = [
    BaseMessage(content="Answer this question: What is in this image?"),
    ImageMessage(image=Path('test/test.png')),
]
```
</details>

<details closed>
<summary>Message Keys</summary>

### Message Keys

>Insert text/image prompts from the `yaml` data into messages.

Keys are replaced with the `yaml` data's text and image prompts during evaluation. Within a message, there are two keys that can be used within a string of a message's content or image:

1. `{{content}}` for the text prompt
2. `{{image}}` for the image prompt

The following example shows the same message lists as the previous examples using message keys:

```python
from evaluation.models.messages.message import UserMessage, SystemMessage, AssistantMessage, ImageMessage, BaseMessage

openai_messages = [
    UserMessage(content="Answer this question: {{content}}", images=["{{image}}"]),
]

deepmind_messages = [
    BaseMessage(content="Answer this question: {{content}}"),
    ImageMessage(image="{{image}}"),
]
```
</details>

<details closed>
<summary>Evaluation</summary>

### Evaluation

Evaluate models on prompts once images are batch generated and automatically linked to the `yaml` data with the `Evaluator`.

The following example evaluates the **OpenAI** model on a batch of binary trees:
```python
from evaluation.evaluator import Evaluator
from evaluation.models.openai import OpenAI
from evaluation.models.messages.message import UserMessage, SystemMessage, AssistantMessage
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY_DEV')

openai = OpenAI(
    api_key=openai_api_key,
)

messages = [UserMessage(content="{{content}}", images=["{{image}}"])]

evaluator = Evaluator()

evaluator.evaluate(
    model=openai,
    messages=messages,
    limit=10,
    yaml_path=Path('data/'),
    yaml_name='binary_tree.yaml',
    csv_path=Path('results/'),
    csv_name='openai.csv'
)
```
</details>
