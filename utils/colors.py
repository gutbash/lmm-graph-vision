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
    
def hex_to_rgb_float(hex_color: str, modifier: int = 0):
    # Remove the '#' character at the beginning if it's there
    hex_color = hex_color.lstrip('#')

    # Convert the hex string to a tuple of integers
    rgb_int = tuple(int(hex_color[i:i+2], 16) + modifier for i in (0, 2, 4))

    # Normalize the RGB values to floats (0.0 to 1.0)
    rgb_float = tuple(round(value / 255, 1) for value in rgb_int)

    return rgb_float