"""
This module contains the constants for the args_parser.py file.
"""

# 1st party imports
from enum import Enum


class RunMode(Enum):
    """Constants for the args_parser.py file."""

    TRAIN_MODE = "train"
    PLAY_MODE = "play"
