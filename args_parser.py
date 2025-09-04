"""
This module is used to parse the arguments for the main.py file.
"""

# 1st party imports
import argparse

# local imports
from args_constants import RunMode


def configure_subparser_for_train(subparser: argparse.ArgumentParser) -> None:
    """
    Configure the subparser for the train mode.

    Args:
        subparser (argparse.ArgumentParser): The subparser to configure.
    """

    # add the config argument
    subparser.add_argument(
        "--config",
        "-c",
        type=str,
        default="neat-config.txt",
        help="Path to the config file (default: neat-config.txt)",
    )

    # add the generations argument
    subparser.add_argument(
        "--generations",
        "-g",
        type=int,
        default=20,
        help="Number of generations to train for",
    )


def configure_subparser_for_play(subparser: argparse.ArgumentParser) -> None:
    """
    Configure the subparser for the play mode.

    Args:
        subparser (argparse.ArgumentParser): The subparser to configure.
    """

    # add the model argument
    subparser.add_argument(
        "--model",
        "-m",
        type=str,
        default="winner.pkl",
        help="Path to the saved model (default: winner.pkl)",
    )

    # add the config argument
    subparser.add_argument(
        "--config",
        "-c",
        type=str,
        default="neat-config.txt",
        help="Path to the config file (default: neat-config.txt)",
    )


def parse_args():
    """
    Parse the arguments for the main.py file.

    Returns:
        argparse.Namespace: The parsed arguments.
    """

    # create the parser
    parser = argparse.ArgumentParser(
        description="Run NEAT algorithm to evolve a neural network to play Flappy Bird."
    )

    # add the subparsers
    subparsers = parser.add_subparsers(
        dest="mode", required=True, help="Choose a mode to run"
    )

    # ---- TRAIN MODE ----
    train_parser = subparsers.add_parser(
        RunMode.TRAIN_MODE.value, help="Train a NEAT model"
    )
    configure_subparser_for_train(train_parser)

    # ---- PLAY MODE ----
    play_parser = subparsers.add_parser("play", help="Play using a trained winner.pkl")
    configure_subparser_for_play(play_parser)

    # parse the args
    args = parser.parse_args()

    return args
