"""
Train a NEAT network to play Flappy Bird.
"""

# 1st party imports
import os
import time
import pickle
import signal
import asyncio
import multiprocessing
from pathlib import Path
from typing import List, Tuple

# 3rd party imports
import neat
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE

# local imports
from args_parser import parse_args
from args_constants import RunMode
from game import (
    Flappy,
    PlayerMode,
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    Score,
    WelcomeMessage,
)


# the folder containg the assets for the game
CURRENT_ROOT = Path(__file__).parent
ASSETS_ROOT = CURRENT_ROOT / "game"

# set the initial window position for this generation
WINDOW_X, WINDOW_Y = -1, -1


class FlappyAI(Flappy):
    """
    This class extends the Flappy class to add AI functionality.
    """

    def __init__(self):
        """
        Initialize the FlappyAI class.
        """
        super().__init__()

    def is_quit_event(self, event: pygame.event.Event) -> bool:
        """
        This function checks if the event is a quit event.

        Args:
            event (pygame.event.Event): The event to check.

        Returns:
            bool: True if the event is a quit event, False otherwise.
        """

        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            return True
        return False

    async def start(self, gid: int, gnome: neat.DefaultGenome, config: neat.Config):
        """
        This function overrides the default start function to add AI functionality.

        Args:
            gnome (neat.DefaultGenome): The genome to evaluate.
            config (neat.Config): The configuration for the NEAT algorithm.

        Returns:
            Tuple[int, float]: The gnome id and fitness.
        """

        # initialize the attributes
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.player = Player(self.config)
        self.welcome_message = WelcomeMessage(self.config)
        self.game_over_message = GameOver(self.config)
        self.pipes = Pipes(self.config)
        self.score = Score(self.config)

        # the start screen is not needed
        # await self.splash()

        # the game over screen is not needed
        # await self.game_over()

        # start the game
        return await self.play(gid, gnome, config)

    async def play(self, gid: int, gnome: neat.DefaultGenome, config: neat.Config):
        """
        This function overrides the default play function to add AI functionality.

        Args:
            gid (int): The gnome id.
            gnome (neat.DefaultGenome): The genome to evaluate.
            config (neat.Config): The configuration for the NEAT algorithm.

        Returns:
            Tuple[int, float]: The gnome id and fitness.
        """

        # reset the game
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        # create the neural network for this game
        neural_net = neat.nn.FeedForwardNetwork.create(gnome, config)

        # initialize the gnome fitness to zero
        gnome.fitness = 0

        # run the game loop 30 times a second (30/fps)
        while True:

            # calculate the distance to the pipe and normalize it
            dist_to_pipe = self.pipes.upper[0].x - self.player.cx

            # calculate the upper pipe's y position
            upper_pipe_cords = self.pipes.upper[0].y + self.pipes.upper[0].h

            # calculate the lower pipe's y position
            lower_pipe_cords = self.pipes.lower[0].y

            # calculate & normalize the gap center
            gap_center = (upper_pipe_cords + lower_pipe_cords) / 2

            # create the environment state
            environment_details = (self.player.cy, dist_to_pipe, gap_center)

            # get the actions
            actions = neural_net.activate((environment_details))

            # return if player collided
            if self.player.collided(self.pipes, self.floor):
                gnome.fitness -= 1  # 1.0 penalty for collision
                return gid, gnome.fitness
            else:
                gnome.fitness += 0.1  # 6.0 reward / second

            # add score if player crossed a pipe
            for pipe in self.pipes.upper:
                if self.player.crossed(pipe):
                    gnome.fitness += 5  # equivalent to 166 seconds of play
                    self.score.add()

            # if the action is over 0.5 (tanH), Jump
            if actions[0] > 0.5:
                self.player.flap()

            # get events
            for event in pygame.event.get():

                # check for quit event
                if self.is_quit_event(event):
                    return gid, gnome.fitness

            # update game objects
            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            # update display
            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()


def get_new_window_cords() -> Tuple[int, int]:
    """
    Get the new window position for the next game.

    Returns:
        Tuple[int, int]: The new window position.
    """

    global WINDOW_X, WINDOW_Y

    # if the window position is -1, reset it to 10, 10
    if WINDOW_X == -1 and WINDOW_Y == -1:
        WINDOW_X, WINDOW_Y = 10, 10
        return WINDOW_X, WINDOW_Y

    # add 300 to x
    new_x = WINDOW_X + 300

    # if new_x is greater than screen width, reset it to 10
    if new_x > 1600:
        new_x = 10
        new_y = WINDOW_Y + 550
    else:
        new_y = WINDOW_Y

    # update the global variables
    WINDOW_X = new_x
    WINDOW_Y = new_y

    # return the new window position
    return WINDOW_X, WINDOW_Y


async def async_eval_gnome(
    gid: int,
    gnome: neat.DefaultGenome,
    config: neat.Config,
    screen_x: int,
    screen_y: int,
):
    """
    Evaluate the fitness of a genome using the Flappy AI game.
    This is an async function to allow for the game to run in the background.

    Args:
        gnome (neat.DefaultGenome): The genome to evaluate.
        config (neat.Config): The configuration for the NEAT algorithm.

    Returns:
        Tuple[int, float]: The gnome id and fitness.
    """

    # set the window position for this game
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{screen_x}, {screen_y}"

    # point the cwd inside the game folder to load all the assests
    os.chdir(ASSETS_ROOT)

    # init the game
    game = FlappyAI()

    # run the game
    gid, fitness = await game.start(gid, gnome, config)

    # reset the cwd to the current directory
    os.chdir(CURRENT_ROOT)

    # return the gnome id and fitness
    return gid, fitness


def eval_gnome(
    gid: int,
    gnome: neat.DefaultGenome,
    config: neat.Config,
    screen_x: int,
    screen_y: int,
):
    """
    Evaluate the fitness of a genome. This is a wrapper function for the eval_gnome function.
    This is used by the NEAT algorithm to evaluate the fitness of a genome.

    Args:
        gnome (neat.DefaultGenome): The genome to evaluate.
        config (neat.Config): The configuration for the NEAT algorithm.

    Returns:
        Tuple[int, float]: The gnome id and fitness.
    """

    return asyncio.run(async_eval_gnome(gid, gnome, config, screen_x, screen_y))


def wrapper_call(args):
    """
    Wrapper function to call eval_gnome.

    Args:
        args: The arguments to pass to eval_gnome.

    Returns:
        Tuple[int, float]: The gnome id and fitness.
    """
    return eval_gnome(*args)


def run_multiple_gnomes(
    gnomes: List[Tuple[int, neat.DefaultGenome]],
    config: neat.Config,
    batch_size: int = 12,
):
    """
    Run the NEAT algorithm to evolve a neural network to play Flappy Bird.
    This function initializes the NEAT configuration, and then uses it to run the evolutionary process.

    Args:
        gnomes (List[Tuple[int, neat.DefaultGenome]]): The list of genomes to evaluate.
        config (neat.Config): The configuration for the NEAT algorithm.
        batch_size (int, optional): The number of genomes to evaluate at once. Defaults to 1.

    Returns:
        None
    """
    global WINDOW_X, WINDOW_Y

    # create a dict with gid: gnome
    gnomes_dict = {gid: gnome for gid, gnome in gnomes}

    # Make sure Ctrl+C kills child processes immediately
    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    process_pool = multiprocessing.Pool(processes=batch_size)
    signal.signal(signal.SIGINT, original_sigint)

    # create the args
    args = [(gid, g, config, *get_new_window_cords()) for gid, g in gnomes]

    try:
        # map the eval_gnome function to the pool
        for result_gid, fitness in process_pool.imap_unordered(wrapper_call, args):

            # Assign fitness
            gnomes_dict[result_gid].fitness = fitness

    # handle keyboard interrupt
    except KeyboardInterrupt:
        process_pool.terminate()
        return

    else:
        # close the pool
        process_pool.close()

    finally:
        # join the pool
        process_pool.join()


def run(config: neat.Config, num_of_generation: int = 20):
    """
    Run the NEAT algorithm to evolve a neural network to play Flappy Bird.
    This function initializes the NEAT configuration, and then uses it to run the evolutionary process.

    Args:
        config (neat.Config): The NEAT configuration.
        num_of_generation (int, optional): Number of generations to run the algorithm. Default is 50.

    Returns:
        None
    """

    # create a population
    population = neat.Population(config)

    # add a stats reporter
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    # map the start time
    start_time = time.time()

    # run the NEAT algorithm
    winner = population.run(run_multiple_gnomes, num_of_generation)

    # print the total time
    print(f"Total time: {(time.time() - start_time):.2f} seconds")

    # save the winner
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":

    # parse the args
    args = parse_args()

    # get the local directory
    local_dir = os.path.dirname(__file__)

    # run the NEAT algorithm
    if args.mode == RunMode.TRAIN_MODE.value:

        # get the path to the config file
        config_path = os.path.join(local_dir, args.config)

        # run the NEAT algorithm
        run(config_path, args.generations)

    elif args.mode == RunMode.PLAY_MODE.value:

        # get the path to the winner
        winner_path = os.path.join(local_dir, args.model)

        # check if the winner exists
        if not os.path.exists(winner_path):
            print("Winner not found")
            exit()

        # get the path to the config file
        config_path = os.path.join(local_dir, args.config)

        # create a neat config
        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        try:

            # load the winner
            with open(winner_path, "rb") as f:
                winner = pickle.load(f)

            # play the winner
            eval_gnome(1, winner, neat_config, 400, 400)

        except Exception as e:
            print(f"Error playing winner: {e}")

    else:
        print("Invalid mode")
        exit()
