
import numpy as np

import pickle
import multiprocessing as mp

try:
    import cv2
    VIDEO = True
except:
    VIDEO = False
    print('Warning: unable to import cv2\n'
          +'Video file cannot be created.')

EXP = 6
MAX_SPAWNS = None
SEED = 0
DRAW = False
ANIMATE = False
FPS = 30
IMAGE_RESCALE = 50

NEW_BOARD_SIZE = (10, 20)

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

def simulate_game(pool, env, decision_maker, save_boards=False):

    if save_boards:
        image_list = []
        env.enable_draw = True
        env.show_images = False

    env.reset()

    if save_boards:
        image_list.append(env.draw())

    while not env.game_over:
        possible_new_boards, _, _ = env.generate_future_boards(None)
        potentials = pool.map(env.generate_future_boards, possible_new_boards)

        secondary_new_boards = np.concatenate([P[0] for P in potentials], axis=0)
        secondary_new_clears = [n for P in potentials for n in P[1]]
        secondary_new_index = [n for n, P in enumerate(potentials) for k in range(P[0].shape[0])]

        secondary_choice = decision_maker(secondary_new_boards, secondary_new_clears)
        choice = secondary_new_index[secondary_choice]

        board_choice = potentials[choice][2]
        env.step(board_choice)

        if save_boards:
            image_list.append(env.draw(alternate_board=env.intermediate_boards[choice]))
            image_list.append(env.draw())
        print('\r{}'.format(env.tetrominos_spawned), end='')
    print()
    if save_boards:
        return image_list, env.tetrominos_spawned
    else:
        return env.tetrominos_spawned

def create_animation(env, images):

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

def main():

    with open('./ecosystems/exp' + str(EXP), 'rb') as inputs:
        ecosystem = pickle.load(inputs)

    best_individual = ecosystem.generation[0]

    ecosystem.environment.max_spawns = MAX_SPAWNS

    if NEW_BOARD_SIZE is not None:
        ecosystem.environment.change_dimensions(*NEW_BOARD_SIZE)
        best_individual.heuristics.alter_dimensions(ecosystem.environment.width, ecosystem.environment.height)

    if ANIMATE and VIDEO:
        ecosystem.environment.enable_draw = True
        ecosystem.environment.show_images = False
    elif DRAW:
        ecosystem.environment.enable_draw = True
        ecosystem.environment.show_images = True
    else:
        ecosystem.environment.enable_draw = False
        ecosystem.environment.show_images = False

    pool = mp.Pool(mp.cpu_count())
    np.random.seed(SEED)
    results = simulate_game(pool, ecosystem.environment, best_individual.evaluate_states, save_boards=ANIMATE)

    if not ANIMATE:
        score = results
    else:
        images = results[0]
        score = results[1]

    print('Seed {}: {}'.format(SEED, score))
    if ANIMATE:
        create_animation(ecosystem.environment, images)

if __name__=='__main__':
    main()

#EOF