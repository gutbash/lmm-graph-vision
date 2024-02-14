from colorlog import ColoredFormatter
from logging import Logger as BaseLogger
import logging

class Logger(BaseLogger):
    
    def __init__(self, name: str, level: int = logging.DEBUG) -> None:
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