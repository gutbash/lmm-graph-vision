"""Contains colors class for the terminal output."""

class Colors:
    """
    Contains colors for the terminal output.
    
    Example
    -------
    ```python
    from utils.colors import Colors
    
    print(f'{Colors.OKGREEN}This is a green message.{Colors.ENDC}')
    ```
    """
    
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'