
import numpy as np

class Tetris_Heuristics():

    def __init__(self, env):

        self.width = env.width
        self.height = env.height
        self.DTYPE = env.DTYPE

        self.board_weights = [k+1 for k in range(self.height-1)]
        self.board_weights.reverse()
        self.board_weights = np.array([self.board_weights
                                       for k in range(self.width-2)],
                                      dtype=self.DTYPE).T

        self.boards = None
        self.true_contours = None
        self.contours = None
        self.wells = None

        self.clears = None
        self.pile_heights = None
        self.altitude_differences = None
        self.weighted_blocks = None
        self.holes = None
        self.row_transitions = None
        self.column_transitions = None
        self.sum_of_all_wells = None
        self.maximum_well_depth = None
        self.bumpiness = None

    def alter_dimensions(self, width, height):

        self.width = width
        self.height = height

        self.board_weights = [k+1 for k in range(self.height-1)]
        self.board_weights.reverse()
        self.board_weights = np.array([self.board_weights
                                       for k in range(self.width-2)],
                                      dtype=self.DTYPE).T

    def determine_values(self, states, clears):

        self.parse_states(states, clears)

        self.calc_pile_heights()
        self.calc_altitude_differences()
        self.calc_weighted_blocks()
        self.calc_holes()
        self.calc_row_transitions()
        self.calc_column_transitions()
        self.calc_wells()
        self.calc_sum_of_all_wells()
        self.calc_maximum_well_depth()
        self.calc_bumpiness()

        values = np.array([
                            self.pile_heights,
                            self.holes,
                            self.clears,
                            self.altitude_differences,
                            self.maximum_well_depth,
                            self.sum_of_all_wells,
                            self.weighted_blocks,
                            self.row_transitions,
                            self.column_transitions,
                            self.bumpiness
                         ], dtype='float32')

        return values

    def parse_states(self, boards, clears):

        self.boards = np.stack(boards)
        self.clears = np.array(clears, dtype=self.DTYPE)

    def calc_pile_heights(self):

        self.true_contours = np.argmax(self.boards, axis=1)
        self.contours = self.height - self.true_contours - 1
        self.pile_heights = np.max(self.contours[:, 1:-1], axis=1)

    def calc_altitude_differences(self):

        self.altitude_differences = (self.pile_heights
                                     - np.min(self.contours[:, 1:-1], axis=1))

    def calc_weighted_blocks(self):

        self.weighted_blocks = np.sum(self.boards[:, 0:-1, 1:-1]
                                      * self.board_weights, axis=(1,2))

    def calc_holes(self):

        self.holes = np.zeros((self.boards.shape[0],), dtype=self.DTYPE)
        for r in range(1, self.height-1):
            self.holes += np.sum(np.logical_and(self.boards[:, r, :] == 0,
                                                np.sum(self.boards[:, 0:r, :],
                                                       axis=1) != 0),
                                 axis=1).astype(self.DTYPE)

    def calc_row_transitions(self):

        self.row_transitions = np.zeros((self.boards.shape[0],),
                                        dtype=self.DTYPE)
        for c in range(0, self.width-1):
            self.row_transitions += np.sum(self.boards[:, :, c]
                                           != self.boards[:, :, c+1],
                                           axis=1).astype(self.DTYPE)

    def calc_column_transitions(self):
        self.column_transitions = np.zeros((self.boards.shape[0],),
                                           dtype=self.DTYPE)
        for r in range(0, self.height-1):
            self.column_transitions += np.sum(self.boards[:, r, :]
                                              != self.boards[:, r+1, :],
                                              axis=1).astype(self.DTYPE)

    def calc_wells(self):

        self.wells = np.zeros((self.boards.shape[0], self.width-2),
                              dtype=self.DTYPE)

        for c in range(1, self.width-1):
            self.wells[:, c-1] = np.maximum(self.contours[:, c-1]
                                            - self.contours[:, c],
                                            self.contours[:, c+1]
                                            - self.contours[:, c])

        self.wells[self.wells < 0] = 0

    def calc_sum_of_all_wells(self):
        self.sum_of_all_wells = np.sum(self.wells, axis=1)

    def calc_maximum_well_depth(self):
        self.maximum_well_depth = np.max(self.wells, axis=1)
    
    def calc_bumpiness(self):
        self.bumpiness = np.sum(np.abs(self.contours[:, 1:-2]
                                       - self.contours[:, 2:-1]),
                                axis=1).astype(self.DTYPE)

def main(N):
    boards, clears, _ = env.generate_future_boards(None)
    for n in range(N):
        state_values = TH.determine_values(boards, np.array(clears))
        print(state_values)

if __name__ == '__main__':
    from tetris import Tetris
    env = Tetris()
    env.id = 0
    TH = Tetris_Heuristics(env)

    N = 1

    main(N)










