
import numpy as np

class Heuristics():
    """Heuristics function for Tetris which help rate the game state.

    Methods
    -------
    alter_dimensions(int, int) --> None
        Alters the dimensions of the assumed game board.
    determine_values(numpy.ndarray, numpy.ndarray) --> numpy.ndarray
        Calculates the ratings for all input boards.
    parse_states(numpy.ndarray, numpy.ndarray) --> None
        Compiles input into desired shape.
    calc_pile_heights(None) -- > None
        Determines the contour of the board, and gets the max height.
    calc_altitude_differences(None) --> None
        Determines the difference between the highest and lowest tile.
    calc_weighted_blocks(None) ---> None
        Deterimes the sum of all tiles, weighted by their height.
    calc_holes(None) --> None
        Determines the number of holes within the board.
    calc_row_transitions(None) --> None
        Determines the number of changes from open to closed tiles row-wise.
    calc_column_transitions(None) --> None
        Determines the number of changes from open to closed tiles column-wise.
    calc_wells(None) --> None
        Determines the depth of each well on the board (hole with no ceiling).
    calc_sum_of_all_wells(None) --> None
        Determines the total sum of all wells on the board.
    calc_maximum_well_depth(None) --> None
        Determines the deepest well on the board.
    calc_bumpiness(None) --> None
        Sums the changes in height from one column to the next.
    """

    def __init__(self, env):
        """Generates a heuristics object to evaluate a given environment.

        Parameters
        ----------
        env : tetris.Tetris
            Working tetris environment.

        Returns
        -------
        None.

        """

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
        """Alters the dimensions of the assumed game board.

        Parameters
        ----------
        width : (int)
            Width of new game board (includes padding).
        height : (int)
            Height of new game board (includes padding).

        Returns
        -------
        None.

        """

        self.width = width
        self.height = height

        self.board_weights = [k+1 for k in range(self.height-1)]
        self.board_weights.reverse()
        self.board_weights = np.array([self.board_weights
                                       for k in range(self.width-2)],
                                      dtype=self.DTYPE).T

    def determine_values(self, states, clears):
        """Calculates the ratings for all input boards.

        Parameters
        ----------
        states : (numpy.ndarray)
            .
        clears : (numpy.ndarray)
            DESCRIPTION.

        Returns
        -------
        values : TYPE
            DESCRIPTION.

        """

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
        values_sqr = values ** 2
        
        full_rating = np.concatenate([values, values_sqr])

        return full_rating

    def parse_states(self, boards, clears):
        """ Loads stack of boards and clears into the heuristic class.

        Parameters
        ----------
        boards : (numpy.ndarray)
            The input boards to evaluate our ratings on.
        clears : (numpy.ndarray)
            The number of line clears achieved by each board.

        Returns
        -------
        None.

        """
        self.boards = np.stack(boards)
        self.clears = clears

    def calc_pile_heights(self):
        """Determines the contour of the board, and gets the max height.
        """

        self.true_contours = np.argmax(self.boards, axis=1)
        self.contours = self.height - self.true_contours - 1
        self.pile_heights = np.max(self.contours[:, 1:-1], axis=1)

    def calc_altitude_differences(self):
        """Determines the difference between the highest and lowest tile.
        """

        self.altitude_differences = (self.pile_heights
                                     - np.min(self.contours[:, 1:-1], axis=1))

    def calc_weighted_blocks(self):
        """Deterimes the sum of all tiles, weighted by their height.
        """

        self.weighted_blocks = np.sum(self.boards[:, 0:-1, 1:-1]
                                      * self.board_weights, axis=(1,2))

    def calc_holes(self):
        """Determines the number of holes within the board.
        """

        self.holes = np.zeros((self.boards.shape[0],), dtype=self.DTYPE)
        for r in range(1, self.height-1):
            self.holes += np.sum(np.logical_and(self.boards[:, r, :] == 0,
                                                np.sum(self.boards[:, 0:r, :],
                                                       axis=1) != 0),
                                 axis=1).astype(self.DTYPE)

    def calc_row_transitions(self):
        """Determines the number of changes from open to closed tiles row-wise.
        """

        self.row_transitions = np.zeros((self.boards.shape[0],),
                                        dtype=self.DTYPE)
        for c in range(0, self.width-1):
            self.row_transitions += np.sum(self.boards[:, :, c]
                                           != self.boards[:, :, c+1],
                                           axis=1).astype(self.DTYPE)

    def calc_column_transitions(self):
        """Equivalent to calc_row_transitions but column-wise.
        """
        self.column_transitions = np.zeros((self.boards.shape[0],),
                                           dtype=self.DTYPE)
        for r in range(0, self.height-1):
            self.column_transitions += np.sum(self.boards[:, r, :]
                                              != self.boards[:, r+1, :],
                                              axis=1).astype(self.DTYPE)

    def calc_wells(self):
        """Determines the depth of each well on the board.
        """

        self.wells = np.zeros((self.boards.shape[0], self.width-2),
                              dtype=self.DTYPE)

        for c in range(1, self.width-1):
            self.wells[:, c-1] = np.maximum(self.contours[:, c-1]
                                            - self.contours[:, c],
                                            self.contours[:, c+1]
                                            - self.contours[:, c])

        self.wells[self.wells < 0] = 0

    def calc_sum_of_all_wells(self):
        """Determines the total sum of all wells on the board.
        """
        self.sum_of_all_wells = np.sum(self.wells, axis=1)

    def calc_maximum_well_depth(self):
        """Determines the deepest well on the board.
        """
        self.maximum_well_depth = np.max(self.wells, axis=1)

    def calc_bumpiness(self):
        """Sums the changes in height from one column to the next.
        """
        self.bumpiness = np.sum(np.abs(self.contours[:, 1:-2]
                                       - self.contours[:, 2:-1]),
                                axis=1).astype(self.DTYPE)

#EOF