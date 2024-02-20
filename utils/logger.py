"""Contains logger class for the terminal output."""

from colorlog import ColoredFormatter
from logging import Logger as BaseLogger
import logging

class Logger(BaseLogger):
    """
    Logger for the terminal output.
    
    Attributes
    ----------
    name : str
        The name of the logger.
    level : int
        The level of the logger.
        
    Example
    -------
    ```python
    from utils.logger import Logger
    
    logger = Logger(__name__)
    logger.debug('This is a debug message.')
    logger.info('This is an info message.')
    logger.warning('This is a warning message.')
    logger.error('This is an error message.')
    logger.critical('This is a critical message.')
    ```
    """
    
    def __init__(self, name: str, level: int = logging.DEBUG) -> None:
        """
        Initialize the logger.
        
        Parameters
        ----------
        name : str
            The name of the logger.
        level : int
            The level of the logger.
        """
        
        super().__init__(name, level)

        # Setting up the logger
        self.setLevel(level)
        self.propagate = False

        # Create a stream handler
        ch = logging.StreamHandler()

        # Create a formatter and set it for the handler
        formatter = ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
        ch.setFormatter(formatter)

        # Add the handler to the logger
        self.addHandler(ch)