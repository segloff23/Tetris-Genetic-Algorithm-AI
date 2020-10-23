
import pickle
import numpy as np

###############################################################################

EXP = 1 # Experiment number to test
N_GAMES = 1 # Number of games to test best individual with
MAX_SPAWNS = None # If None, game ends when agent loses
NEW_BOARD_SIZE = (10, 20) # If None, use ecosystems preset environment
USE_POOL = True # Make use of multiproccessing pool

###############################################################################
def main():
    """Test the best individual from a certain ecosystem.
    """

    with open('./ecosystems/exp' + str(EXP), 'rb') as inputs:
        ecosystem = pickle.load(inputs)

    best_individual = ecosystem.generation[0]

    if NEW_BOARD_SIZE is not None:
        ecosystem.env.change_dimensions(*NEW_BOARD_SIZE)
        best_individual.heuristics.alter_dimensions(ecosystem.env.width, ecosystem.env.height)

    ecosystem.env.max_spawns=MAX_SPAWNS

    if USE_POOL:
        from multiprocessing import Pool, cpu_count
        pool = Pool(cpu_count())
    else:
        pool = None

    for n in range(N_GAMES):
        np.random.seed(n+1)
        score = ecosystem.env.simulate(best_individual.evaluate_states, pool=pool)
        print('Seed {}: {}'.format(n+1, score))

if __name__=='__main__':
    main()

#EOF