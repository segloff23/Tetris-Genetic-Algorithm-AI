
import numpy as np
import dill as pickle

try:
    import cv2
    VIDEO = True
except:
    VIDEO = False
    print('Warning: unable to import cv2\n'
          +'Video file cannot be created.')

###############################################################################

EXP = 3 # Experiment to laod ecosystem from
MAX_SPAWNS = None # If None, game ends only if agent loses
SEED = 1 # Specifiy seed to illustrate game on
NEW_BOARD_SIZE = (10, 20) # If None, use ecosystem preset environment
USE_POOL = True # Make use of multiprocessing

# Create sequence of figures illustrating game
# WARNING: To prevent too many figures from being created,
#           do not set MAX_SPAWNS to equal None
DRAW = False

# Create video file using mp4 codec of the given game
# WARNING: To prevent excessive video lengths,
#           do not set MAX_SPAWNS to equal None
ANIMATE = False
FPS = 30 # Each move is given by 1 or 2 frames, depending on line clears
IMAGE_RESCALE = 50 # Scale up board from smaller simulation size

###############################################################################

# Customize tetromino colors for video creation if desired
COLORS = (
            (0,     0,   0),
            (255,   0,   0),
            (0,   128,   0),
            (0,   0,   255),
            (255, 128,   0),
            (255, 255,   0),
            (128, 0  , 255),
            (0,   255, 255),
            (255, 255, 255)
         )

###############################################################################

def simulate_and_save(env, decision_maker, save_boards=False, pool=None):
    """Custom simulator to allow saving of each board for animation.

    Parameters
    ----------
    env : (tetris.Tetris)
        The Tetris environment the game is played within.
    decision_maker : (function handle)
        Function used to decide which board is optimal.
        Requires two inputs, boards and clears, which are numpy.ndarry.
        Must ouput a single index, describing the optimal board
    save_boards : (bool), optional
        If True, a list of boards will be kept each step.
        The default is False.
    pool : (Pool.pool) or (None), optional
        Pool used to perform multiprocessing on.
        The default is None.

    Returns
    -------
    image_list : (list of numpy.ndarray)
        Returned only if save_boards is True. It is used to generate
        a video from the sequence
    tetrominos_spawned : (int)
        The number of tetrominos spawned over the course of the game.
    """

    env.reset()
    if save_boards:
        image_list = [env.draw()]

    while not env.game_over:
        potential_boards, _ = env.generate_future_boards(None)

        if pool is not None:

            pool_output = pool.map(env.generate_future_boards,
                                   potential_boards)
            potential_next_boards = np.concatenate([P[0] for P in pool_output],
                                                   axis=0)
            potential_next_clears = np.concatenate([P[1] for P in pool_output])
            potential_board_indexes = [n for n, P in enumerate(pool_output)
                                       for k in range(P[0].shape[0])]
        else:
            potential_next_boards = []
            potential_next_clears = []
            potential_board_indexes = []
            for n, board in enumerate(potential_boards):
                (next_boards,
                 next_line_clears) = env.generate_future_boards(board)
                potential_next_boards.append(next_boards)
                potential_next_clears.append(next_line_clears)
                potential_board_indexes.extend([n]*len(next_boards))

            potential_next_boards = np.concatenate(potential_next_boards)
            potential_next_clears = np.concatenate(potential_next_clears)

        next_state_of_choice_index = decision_maker(potential_next_boards,
                                                    potential_next_clears)
        state_of_choice_index = potential_board_indexes[
                                                next_state_of_choice_index]
        env.step(potential_boards[state_of_choice_index])

        if save_boards:
            if potential_next_clears[state_of_choice_index]:
                transition = env.intermediate_boards[state_of_choice_index]
                image_list.append(env.draw(alternate_board=transition))
            image_list.append(env.draw())
        if env.tetrominos_spawned % 100 == 0:
            print('\rStep {}'.format(env.tetrominos_spawned), end='')
    print()

    if save_boards:
        return image_list, env.tetrominos_spawned
    else:
        return env.tetrominos_spawned

def create_animation(env, images):
    """Generate a video of a Tetris game from a list of numpy.ndarray 'images.'

    Parameters
    ----------
    env : (tetris.Tetris)
        The environment the game used to generate the images.
    images : (list of numpy.ndarray)
        The images to be compiled into a video.

    Returns
    -------
    None.

    """

    # We keep the borders, but remove top padding
    og_width =  env.width
    og_height = env.height-4

    width =  og_width * IMAGE_RESCALE
    height = og_height * IMAGE_RESCALE

    file_name = './videos/video_exp' + str(EXP) + '_seed' + str(SEED) + '.avi'
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(file_name, codec, float(FPS), (width, height))

    for raw_image in images:

        image = np.ones((height, width, 3), dtype='uint8')

        for r in range(og_height):
            for c in range(og_width):

                rgb = COLORS[raw_image[r, c]]

                r0 = r*IMAGE_RESCALE
                r1 = (r+1)*IMAGE_RESCALE
                c0 = c*IMAGE_RESCALE
                c1 = (c+1)*IMAGE_RESCALE

                image[r0:r1, c0:c1, :] = np.array(rgb, dtype='uint8')
        writer.write(image)

    writer.release()
    writer=None
    print('Video saved to ' + file_name)

def main():
    """Draw, animate, or just test a single game of a single individual.
    """

    with open('./ecosystems/exp' + str(EXP), 'rb') as inputs:
        ecosystem = pickle.load(inputs)

    best_individual = ecosystem.generation[0]

    ecosystem.env.max_spawns = MAX_SPAWNS
    if NEW_BOARD_SIZE is not None:
        ecosystem.env.change_dimensions(*NEW_BOARD_SIZE)
        ecosystem.heuristics.alter_dimensions(ecosystem.env.width,
                                                    ecosystem.env.height)

    if ANIMATE and VIDEO:
        ecosystem.env.enable_draw = True
        ecosystem.env.show_images = False
    elif DRAW:
        ecosystem.env.enable_draw = True
        ecosystem.env.show_images = True
    else:
        ecosystem.env.enable_draw = False
        ecosystem.env.show_images = False

    if USE_POOL:
        from multiprocessing import Pool, cpu_count
        pool = Pool(cpu_count())
    else:
        pool=None

    decision_maker = lambda x,y : (best_individual.evaluate_states(
                                ecosystem.heuristics.determine_values(x, y)))

    np.random.seed(SEED)
    results = simulate_and_save(ecosystem.env, decision_maker,
                                save_boards=ANIMATE, pool=pool)
    if not ANIMATE:
        score = results
        print('Seed {}: {}'.format(SEED, score))
    else:
        images = results[0]
        score = results[1]
        create_animation(ecosystem.env, images)

if __name__=='__main__':
    main()

#EOF