# lmm-graph-tree-vqa

How well do the **GPT-4V** and **Gemini Pro Vision** models perform **Visual Question Answering (VQA)** on **Data Structures**?

There is no dataset for **VQA** on graph and tree data structures in previous work, so we must create one. We create a standard, repeatable process for selecting and obtaining **VQA** tasks that fall under a certain criteria.



## Quickstart

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