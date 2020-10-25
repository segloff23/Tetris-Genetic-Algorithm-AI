
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

EXP = 1 # Experiment to laod ecosystem from
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

    decision_maker = lambda B, C : (
                ecosystem.evaluator(*best_individual.model_values,
                               ecosystem.heuristics.determine_values(B, C)))

    np.random.seed(SEED)
    results = ecosystem.env.simulate(decision_maker, pool=pool,
                                     print_steps=True, log_images=ANIMATE)
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