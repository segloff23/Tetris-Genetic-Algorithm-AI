
import time
import dill as pickle
#import pickle
import os
import sys
sys.path.append('.') # Seems to prevent issue with multiproccessing

from tetris import Tetris
from heuristics import Heuristics
from ecosystem import Ecosystem

###############################################################################

# GAME VARIABLES
WIDTH = 6 # No padding
HEIGHT = 12 # No padding
MAX_SPAWNS = None # If None, games last until agent loses
ACTIVE_TETROMINOS = None # If None, all tetrominos are used

# EVOLUTION VARIABLES
MODEL_NAME = 'model_wed'
GEN_SIZE = 10 # Minimum size is 2
GAME_SIZE = 1 # Number of games to test fitness over
MUT_PROB = 0.2 # Probability of an individual gene mutating
N_EPISODES = 1 # Number of generations to iterate through

# DATA VARIABLES
EXP = 2 # Experiment number
LOAD = False # Load from file given be EXP or not
SAVE_FREQ = 5 * 60 # How often to save over results in seconds
USE_POOL = True # Make use of multiprocessing for simulations
SEED = 1 # Initialize with a specific seed to replicate results
LOGGING = False # Log data for tensorboard live visualization

###############################################################################

def load_ecosystem():
    """Loads an ecosystem from a given file

    Returns
    -------
    ecosystem : (ecosystem.Ecosystem)
        The ecosystem class which evolution will take place in.
    """

    with open('./ecosystems/exp' + str(EXP), 'rb') as inputs:
        ecosystem = pickle.load(inputs)
    print('Ecosystem created.')

    return ecosystem

def validate_save_file():
    """Verifies the user wants to use the given experiment number.

    Returns
    -------
    bool
        True to continue evolution, False to cancel.

    """

    experiments = [int(f.name[3::]) for f in os.scandir('./ecosystems')]
    if EXP in experiments:
        s = 'Warning: An ecosystem exists for experiment #{}.\n'
        print(s.format(EXP) +'This file will be overwritten.')

        while True:
            response = input('Continue? (y/n) : ')
            if response == 'n':
                return False
            elif response == 'y':
                print()
                return True
            else:
                print('Invalid input. Please try again.', end='')
    else:
        return True

def main():
    """Evolve an ecosystem to play Tetris with the given constant parameters.
    """

    if SEED is not None:
        from numpy.random import seed
        seed(SEED)

    env = Tetris(width=WIDTH, height=HEIGHT, max_spawns=MAX_SPAWNS,
                 active_tetrominos=ACTIVE_TETROMINOS)
    heuristics = Heuristics(env)

    print()
    if LOAD:
        ecosystem = load_ecosystem()
    else:
        valid_save = validate_save_file()
        if not valid_save:
            return
        else:
            from importlib import import_module
            from ecosystem import generate_model

            model_generator = import_module('models.' + MODEL_NAME)
            model_settings, evaluator = model_generator.get_model_settings()
            model = generate_model(*model_settings)

            ecosystem = Ecosystem(env, heuristics, GEN_SIZE, model, evaluator,
                                  MODEL_NAME)
            print('Ecosystem created.')

    if LOGGING:
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            writer = SummaryWriter('./logs/exp' + str(EXP))
            disable_logging = False
        except:
            print('torch.utils.tensorboard not found, logging disabled.')
            disable_logging = True

    if USE_POOL:
        from multiprocessing import Pool, cpu_count
        pool = Pool(cpu_count())
    else:
        pool = None

    s ='Saving to /ecosystems/exp{} every {} seconds.'
    print(s.format(EXP, SAVE_FREQ))
    print('Beginning evolution.')
    print()

    report_string = 'Episode {}: {:7.2f} / {:<7.2f}'
    start = time.time()
    for ep in range(N_EPISODES):

        mean_score, elite_mean_score = ecosystem.evolve(GAME_SIZE, GEN_SIZE,
                                                        MUT_PROB, pool=pool)

        if LOGGING and not disable_logging:
            writer.add_scalar('Mean_Score', mean_score, ep+1)
            writer.add_scalar('Elite_Mean_Score', elite_mean_score, ep+1)

        print(report_string.format(ecosystem.generation_number,
                                   mean_score, elite_mean_score))

        if (ep == 0) or (time.time() - start) >= SAVE_FREQ:
            with open('./ecosystems/exp' + str(EXP), 'wb') as output:
                pickle.dump(ecosystem, output, -1)
            print('Ecosystem saved.')
            start = time.time()
        print()

if __name__ == '__main__':
    main()

#EOF