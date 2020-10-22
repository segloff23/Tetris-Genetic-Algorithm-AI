
import numpy as np

import time
import pickle
import multiprocessing as mp

from another_tetris import Tetris
from tetris import simulate_game
from tetris_heuristics import Tetris_Heuristics
from ecosystem import Ecosystem

###############################################################################

EXP = 7
N_GAMES = 0
MAX_SPAWNS = None

###############################################################################
def main():
    
    with open('./ecosystems/exp' + str(EXP), 'rb') as inputs:
        ecosystem = pickle.load(inputs)
    

    best_individual = ecosystem.generation[0]
    print(best_individual.weights)

    ecosystem.environment.max_spawns=MAX_SPAWNS
    pool = mp.Pool(mp.cpu_count())
    
    for n in range(N_GAMES):
        np.random.seed(n+1)
        score = simulate_game(pool,
                              ecosystem.environment,
                              best_individual.evaluate_states)
        print('Game {}: {}'.format(n+1, score))

if __name__=='__main__':
    main()

#EOF