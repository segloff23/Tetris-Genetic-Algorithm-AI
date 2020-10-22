
import time
import pickle
import multiprocessing as mp
import os

from another_tetris import Tetris
from tetris import simulate_game
from tetris_heuristics import Tetris_Heuristics
from ecosystem import Ecosystem

try:
    from torch.utils.tensorboard import SummaryWriter
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    LOGGING = True
except:
    LOGGING = False

###############################################################################

# GAME VARIABLES
WIDTH = 6
HEIGHT = 12
MAX_SPAWNS = None
ACTIVE_TETROMINOS = None

# EVOLUTION VARIABLES
GEN_SIZE = 100
GAME_SIZE = 5
MUT_PROB = 0.2
N_EPISODES = 100000

# DATA VARIABLES
EXP = 7
SAVE_FREQ = 5 * 60

###############################################################################

def main():

    experiments = [int(f.name[3::]) for f in os.scandir('./ecosystems')]

    if EXP in experiments:
        print('Warning: An ecosystem exists for experiment #{}.\n'.format(EXP)
              +'This file will be overwritten.'
              )
        valid = False
        while not valid:
            response = input('Continue? (y/n) : ')
            if response == 'n':
                return
            elif response == 'y':
                valid = True
            else:
                print('Invalid input. Please try again.', end='')

    if LOGGING:
        writer = SummaryWriter('./logs/exp' + str(EXP))

    pool = mp.Pool(mp.cpu_count())

    env = Tetris(width=WIDTH, height=HEIGHT, max_spawns=MAX_SPAWNS,
                 active_tetrominos=ACTIVE_TETROMINOS)
    heuristics = Tetris_Heuristics(env)
    ecosystem = Ecosystem(env, simulate_game, heuristics, GEN_SIZE)

    print()
    print('Ecosystem created.')
    print('Saving to /ecosystems/exp{} every {} seconds.'.format(EXP, SAVE_FREQ))
    print('Beginning evolution.')
    print()
    start = time.time()
    for ep in range(N_EPISODES):

        mean_score, elite_mean_score = ecosystem.evolve(pool, GAME_SIZE,
                                                        GEN_SIZE, MUT_PROB)

        if LOGGING:
            writer.add_scalar('Mean_Score', mean_score, ep+1)
            writer.add_scalar('Elite_Mean_Score', elite_mean_score, ep+1)

        print('Episode {}: {:7.2f} / {:<7.2f}'.format(ep+1,
                                                      mean_score,
                                                      elite_mean_score))

        if (ep == 0) or (time.time() - start) >= SAVE_FREQ:
            with open('./ecosystems/exp' + str(EXP), 'wb') as output:
                pickle.dump(ecosystem, output, -1)
            print('Ecosystem saved.')
            start = time.time()

        print()

if __name__ == '__main__':
    main()

#EOF