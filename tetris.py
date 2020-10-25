
import numpy as np
import matplotlib.pyplot as plt

class Tetris():
    """Environment for playing tetris with state-based decisions.

    Playing a game within this environment with no lookahead
    consists of the following code loop, where env = Tetris(),
    and the player is represented by choose_a_board():

        env.reset()
        while not env.game_over:
            (potential_boards,
             lines_cleared, _) = env.generate_future_boards(None)
            board_of_choice_index = choose_a_board(potential_boards,
                                                   lines_cleared)
            env.step(potential_boards[board_of_choice_index])

    To play a game with lookahead, use the simulate method,
    without or without a Pool object to make use of multiprocessing

    Methods
    -------
    initialize_board(int, int, int) --> (numpy.ndarray)
        Returns an array representing the Tetris board.
    change_dimensions(int, int) --> (None)
        Alters the dimensions of the game board and resets the game.
    spawn_tetromino() --> (None)
        Updates queue of tetrominos, and increments tetrominos_spawned.
    generate_future_boards(numpy.ndarray) --> (List of numpy.ndarray,
                                               numpy.ndarray,
                                               numpy.ndarray)
        Generates board possibilities reachable by a starting board.
    depth_first_search(numpy.ndarray, int, int) --> (List of numpy.ndarray)
        Employs a DFS to locate reachable new board for a given tetromino.
    depth_first_search_recursive_call(numpy.ndarray,
                                      int, int,
                                      complex, set,
                                      list of numpy.ndarray) --> (None)
        The DFS recursion function called on each new node.
    update_all_boards(numpy.ndarray) --> (numpy.ndarray, numpy.ndarray)
        Clears the completed Tetris lines of a stack of game boards.
    step(numpy.ndarray) --> (None)
        Advances the game one step to the provided game board.
    update_game_over_status(None) --> (None)
        Updates the status of game_over using current_board.
    reset(None) --> (None)
        Resets the game state to the starting point.
    draw(np.ndarray) --> (None)
        Generates an image of a game board.

    """

    DTYPE = 'uint8'

    # 0 : O Piece
    # 1 : I Piece
    # 2 : S Piece
    # 3 : Z Piece
    # 4 : L Piece
    # 5 : J Piece
    # 6 : T Piece

    TETROMINOS = (
                    (
              np.array(((0, 0, 0, 0),
                        (0, 1, 1, 0),
                        (0, 1, 1, 0),
                        (0, 0, 0, 0)), dtype=DTYPE),
                    ),

                    (
               np.array(((0, 0, 0, 0),
                         (1, 1, 1, 1),
                         (0, 0, 0, 0),
                         (0, 0, 0, 0)), dtype=DTYPE),

               np.array(((0, 0, 1, 0),
                         (0, 0, 1, 0),
                         (0, 0, 1, 0),
                         (0, 0, 1, 0)), dtype=DTYPE)
                    ),

                    (
               np.array(((0, 0, 0, 0),
                         (0, 0, 1, 1),
                         (0, 1, 1, 0),
                         (0, 0, 0, 0)), dtype=DTYPE),

               np.array(((0, 0, 1, 0),
                         (0, 0, 1, 1),
                         (0, 0, 0, 1),
                         (0, 0, 0, 0)), dtype=DTYPE)
                    ),

                    (
               np.array(((0, 0, 0, 0),
                         (0, 1, 1, 0),
                         (0, 0, 1, 1),
                         (0, 0, 0, 0)), dtype=DTYPE),

               np.array(((0, 0, 0, 1),
                         (0, 0, 1, 1),
                         (0, 0, 1, 0),
                         (0, 0, 0, 0)), dtype=DTYPE)
                    ),

                    (
               np.array(((0, 0, 0, 0),
                         (0, 1, 1, 1),
                         (0, 1, 0, 0),
                         (0, 0, 0, 0)), dtype=DTYPE),

               np.array(((0, 0, 1, 0),
                         (0, 0, 1, 0),
                         (0, 0, 1, 1),
                         (0, 0, 0, 0)), dtype=DTYPE),

               np.array(((0, 0, 0, 1),
                         (0, 1, 1, 1),
                         (0, 0, 0, 0),
                         (0, 0, 0, 0)), dtype=DTYPE),

               np.array(((0, 1, 1, 0),
                         (0, 0, 1, 0),
                         (0, 0, 1, 0),
                         (0, 0, 0, 0)), dtype=DTYPE)
                    ),

                    (
               np.array(((0, 0, 0, 0),
                         (0, 1, 1, 1),
                         (0, 0, 0, 1),
                         (0, 0, 0, 0)), dtype=DTYPE),

               np.array(((0, 0, 1, 1),
                         (0, 0, 1, 0),
                         (0, 0, 1, 0),
                         (0, 0, 0, 0)), dtype=DTYPE),

               np.array(((0, 1, 0, 0),
                         (0, 1, 1, 1),
                         (0, 0, 0, 0),
                         (0, 0, 0, 0)), dtype=DTYPE),

               np.array(((0, 0, 1, 0),
                         (0, 0, 1, 0),
                         (0, 1, 1, 0),
                         (0, 0, 0, 0)), dtype=DTYPE)
                    ),

                    (
               np.array(((0, 0, 0, 0),
                         (0, 1, 1, 1),
                         (0, 0, 1, 0),
                         (0, 0, 0, 0)), dtype=DTYPE),

               np.array(((0, 0, 1, 0),
                         (0, 0, 1, 1),
                         (0, 0, 1, 0),
                         (0, 0, 0, 0)), dtype=DTYPE),

               np.array(((0, 0, 1, 0),
                         (0, 1, 1, 1),
                         (0, 0, 0, 0),
                         (0, 0, 0, 0)), dtype=DTYPE),

               np.array(((0, 0, 1, 0),
                         (0, 1, 1, 0),
                         (0, 0, 1, 0),
                         (0, 0, 0, 0)), dtype=DTYPE)
                    ),
                 )

    RC_ITERATOR = [[
                    [(r, c) for r in range(4) for c in range(4) if t[r][c]]
                            for t in T]
                            for T in TETROMINOS]

    OR_ITERATOR = [range(len(T)) for T in TETROMINOS]

    def __init__(self, width=10, height=20,
                 max_spawns=None, active_tetrominos=None, enable_draw=False,
                 show_images=False):
        """Environment for playing tetris with state-based decisions.

        Parameters
        ----------
        width : (int), optional
            The width of the game board, excluding borders.
            The default is 10.
        height : (int), optional
            The height of the game board, excluding borders.
            The default is 20.
        max_spawns : (int) or (None), optional
            The maximum number of tetrominos that can be spawned before
            the game automatically ends. If None, the game will not end
            until the player loses.
            The default is None.
        active_tetrominos : (list of int) or (None), optional
            The indices [0-6] of tetrominos the game will use. If None,
            all tetrominos are considered active.
            The default is None.
        enable_draw : (bool), optional
            If True, draw() will be called upon initialization and each time
            step() is called.
            The default is False.
        show_images : (bool), optional
            If True, draw() will create a new figure of the current game board
            upon initialization and each time step() is called.
            The defaultis False.

        Returns
        -------
        None.

        """

        self.width = width + 1 + 1
        self.height = height + 4 + 1
        self.max_spawns = max_spawns
        self.active_tetrominos = active_tetrominos
        self.enable_draw = enable_draw
        self.show_images = show_images

        if self.active_tetrominos is None:
            self.active_tetrominos = list(range(len(self.TETROMINOS)))
        self.active_tetrominos_count = len(self.active_tetrominos)

        self.starting_board = self.initialize_board(
                                                self.width,
                                                self.height,
                                                self.active_tetrominos_count)
        self.current_board = self.starting_board.copy()
        self.intermediate_boards = None

        self.current_tetromino = None
        self.next_tetromino = None

        self.tetrominos_spawned = 0
        self.spawn_tetromino()

        self.game_over = False

        if self.enable_draw:
            self.draw()

    def initialize_board(self, w, h, N):
        """Returns an array representing the Tetris board.

        Parameters
        ----------
        w : (int)
            Width of the board, including borders.
        h : (int)
            Height of the board, including borders.
        N : (int)
            Fill value for the borders.

        Returns
        -------
        board : (numpy.ndarray)
            Array filled by zeros with left, right, and bottom borders
            filled by N.

        """

        board = np.zeros((h, w), dtype=self.DTYPE)
        board[:, 0].fill(N+1)
        board[:, -1].fill(N+1)
        board[-1, :].fill(N+1)

        return board

    def change_dimensions(self, new_width, new_height):
        """Alters the dimensions of the game board and resets the game.

        Parameters
        ----------
        new_width : (int)
            The new width of the game board, excluding borders.
        new_height : (int)
            The new height of the game board, excluding borders.

        Returns
        -------
        None.

        """

        self.width = new_width + 1 + 1
        self.height = new_height + 4 + 1
        self.starting_board = self.initialize_board(
                                                self.width,
                                                self.height,
                                                self.active_tetrominos_count)
        self.reset()

    def spawn_tetromino(self):
        """Updates queue of tetrominos, and increments tetrominos_spawned.

        Returns
        -------
        None.

        """

        self.tetrominos_spawned += 1
        if self.next_tetromino is not None:
            self.current_tetromino = self.next_tetromino
        else:
            self.current_tetromino = np.random.choice(self.active_tetrominos)

        self.next_tetromino = np.random.choice(self.active_tetrominos)

    def generate_future_boards(self, lookahead_board):
        """Generates board possibilities reachable by a starting board.

        Parameters
        ----------
        lookahead_board : (numpy.ndarray) or (None)
            The starting board to explore. If None is provided, defaults
            to current_board and uses current_tetromino. Otherwise,
            the provbided board is used along with next_tetromino

        Returns
        -------
        updated_future_boards : (list of numpy.ndarray)
            All reachable boards given a starting board and tetromino.
        future_boards_lines_cleared : (numpy.ndarray)
            The number of lines cleared by reaching the corresponding board
            in updated_future_boards.

        """

        if lookahead_board is not None:
            starting_board = lookahead_board
            tetromino_id = self.next_tetromino
        else:
            starting_board = self.current_board
            tetromino_id = self.current_tetromino

        future_boards = self.depth_first_search(starting_board, tetromino_id)

        # For animation purposes, we may want to save the boards
        # before lines have been cleared
        if lookahead_board is None:
            self.intermediate_boards = future_boards

        (updated_future_boards,
        future_boards_lines_cleared) = self.update_all_boards(
                                                    np.array(future_boards))

        return (updated_future_boards, future_boards_lines_cleared)

    def depth_first_search(self, board, tetromino_id):
        """Explore all possible states reachable by a given tetromino.

        Parameters
        ----------
        board : (numpy.ndarray)
            The initial board the tetromino will explore.
        tetromino_id : (int)
            The index corresponding to the exploring tetromino.

        Returns
        -------
        boards_created : (list of numpy.ndarray)
            The list of boards created by all possible piece placements.

        """

        y = 0
        while not np.any(board[y+1, 1:-1]):
            y += 1

        y = max([0, y-4])

        source_node = (1+1j*y, 0)
        visited_nodes = set()
        boards_created = []

        self.depth_first_search_recursive_call(
                                        board, np.array(board, dtype='bool'),
                                        tetromino_id,
                                        source_node, visited_nodes,
                                        boards_created)

        return boards_created

    def depth_first_search_recursive_call(self, real_board, board,
                                          tetromino_id,
                                          current_node, visited_nodes,
                                          boards_created):
        """ Recursive call for implementation of DFS.

        Parameters
        ----------
        real_board : (numpy.ndarray)
            The original game board explored by the DFS.
        board : (numpy.ndarray)
            The original game board represented as booleans, used
            for faster checking by the DFS.
        tetromino_id : (int)
            The index corresponding to the exploring tetromino.
        current_node : (tuple: complex, int)
            The coordinates and current orientation id of the tetromino.
            The real part is the column index, the imaginary the row index.
        visited_nodes : (set)
            Set of nodes DFS has visited. Nodes have the form of current_node.
        boards_created : (list of numpy.ndarray)
            The list of boards created by all possible piece placements.

        Returns
        -------
        None.

        """

        visited_nodes.add(current_node)

        node_coord, node_orient = current_node
        x, y = int(node_coord.real), int(node_coord.imag)

        nodes_to_explore = set()

        test_node = (node_coord+1j, node_orient)
        if test_node not in visited_nodes:
            connected = True
            for r, c in self.RC_ITERATOR[tetromino_id][node_orient]:
                if board[y+1+r % self.height, x+c % self.width]:
                    connected = False
                    break
            if connected:
                nodes_to_explore.add(test_node)
            else:
                new_board = real_board.copy()
                fill_val = tetromino_id + 1
                for r, c in self.RC_ITERATOR[tetromino_id][node_orient]:
                    new_board[y+r % self.height, x+c % self.width] = fill_val
                boards_created += [new_board]

        test_node = (node_coord+1, node_orient)
        if test_node not in visited_nodes:
            connected = True
            for r, c in self.RC_ITERATOR[tetromino_id][node_orient]:
                if board[y+r % self.height, x+1+c % self.width]:
                    connected = False
                    break
            if connected:
                nodes_to_explore.add(test_node)

        test_node = (node_coord-1, node_orient)
        if test_node not in visited_nodes:
            connected = True
            for r, c in self.RC_ITERATOR[tetromino_id][node_orient]:
                if board[y+r % self.height, x-1+c % self.width]:
                    connected = False
                    break
            if connected:
                nodes_to_explore.add(test_node)

        for alt_orient in self.OR_ITERATOR[tetromino_id]:
            test_node = (node_coord, alt_orient)
            if test_node not in visited_nodes:
                connected = True
                for r, c in self.RC_ITERATOR[tetromino_id][alt_orient]:
                    if board[y+r % self.height, x+c % self.width]:
                        connected = False
                        break
                if connected:
                    nodes_to_explore.add(test_node)

        for new_node in nodes_to_explore:
            if new_node not in visited_nodes:
                self.depth_first_search_recursive_call(real_board, board,
                                                       tetromino_id,
                                                       new_node, visited_nodes,
                                                       boards_created)

    def update_all_boards(self, boards):
        """Clears the completed Tetris lines of a stack of game boards.

        Parameters
        ----------
        boards : (numpy.ndarray)
            A matrix of size (N,h,w) consiting of a stack of game boards.

        Returns
        -------
        updated_boards : (numpy.ndarray)
            A matrix of size (N,h,w) consiting of a stack of game boards
            after completed Tetris lines have been cleared.
        lines_cleared : (numpy.ndarray)
            A vector of size (N,) consisting of the number of lines cleared
            for each of the N game boards updated.

        """

        mask = np.all(boards, axis=2)
        lines_cleared = np.sum(mask, axis=1) - 1

        np.logical_not(mask, out=mask)
        updated_boards = np.tile(self.starting_board, (boards.shape[0], 1, 1))
        for b in range(boards.shape[0]):
            updated_boards[b, lines_cleared[b]:-1, :] = boards[b, mask[b], :]

        return updated_boards, lines_cleared

    def step(self, new_board):
        """Advances the game one step to the provided game board.

        Parameters
        ----------
        new_board : (numpy arary)
            A valid game board. step() does not verify if the provided board
            is reachable from the current one.

        Returns
        -------
        None.

        """

        self.current_board = new_board
        self.update_game_over_status()

        if self.enable_draw:
            self.draw()

        if not self.game_over:
            self.spawn_tetromino()

    def update_game_over_status(self):
        """Updates the status of game_over using current_board.

        Returns
        -------
        None.

        """

        if np.any(self.current_board[5, 1:-1]):
            self.game_over = True
        elif self.max_spawns is not None:
            if self.tetrominos_spawned >= self.max_spawns:
                self.game_over = True

    def reset(self):
        """Resets the game state to the starting point.

        Returns
        -------
        None.

        """

        self.current_board = self.starting_board.copy()

        self.current_tetromino = None
        self.next_tetromino = None

        self.tetrominos_spawned = 0
        self.spawn_tetromino()

        self.game_over = False

    def draw(self, alternate_board=None):
        """ Generates an image of a game board.

        Parameters
        ----------
        alternate_board : (numpy.ndarray) or (None), optional
            Construct an image from the given board. If None,
            current_board is used to construct the image.
            The default is None.

        Returns
        -------
        image : (numpy.ndarray)
            A grayscale image of the gameboard.

        """

        if alternate_board is not None:
            image = self.active_tetrominos_count+1 - alternate_board
        else:
            image = self.active_tetrominos_count+1 - self.current_board

        if self.show_images:
            plt.figure()
            plt.imshow(image[4::, :], cmap='gray',
                       vmin=0, vmax=self.active_tetrominos_count+1)

        return image[4::, :]

    def simulate(self, decision_maker, pool=None, print_steps=False,
                 log_images=False):
        if pool is not None:
            return simulate_with_pool(self, decision_maker, pool, print_steps,
                                      log_images)
        else:
            return simulate_without_pool(self, decision_maker, print_steps,
                                         log_images)

def simulate_with_pool(env, decision_maker, pool, print_steps, log_images):
    """Using multiprocessing, simulates a game of Tetris with lookahead.

    Parameters
    ----------
    pool : (multiprocessing.pool.Pool)
        Pool object to determine potential boards with.
    env : (Tetris)
        A Tetris environment to simulate a game within.
    decision_maker : (function handle)
        A function to determine which state to pursue. Must take in
        two inputs:
            boards (numpy.ndarray)
            line_clears (numpy.ndarray)
        The function then returns the index corresponding to the
        desired state to pursue.

    Returns
    -------
    env.tetrominos_spawned : (int)
        The number of tetrominos spawned before the game ended,
        equivalent to the number of steps taken.

    """

    env.reset()
    if log_images:
        image_list = [env.draw()]
    while not env.game_over:

        potential_boards, _ = env.generate_future_boards(None)
        pool_output = pool.map(env.generate_future_boards, potential_boards)

        potential_next_boards = np.concatenate([P[0] for P in pool_output],
                                               axis=0)
        potential_next_clears = np.concatenate([P[1] for P in pool_output])

        potential_board_indexes = [n for n, P in enumerate(pool_output)
                                   for k in range(P[0].shape[0])]

        next_state_of_choice_index = decision_maker(potential_next_boards,
                                                    potential_next_clears)
        state_of_choice_index = potential_board_indexes[
                                                next_state_of_choice_index]
        env.step(potential_boards[state_of_choice_index])
        if log_images:
            if potential_next_clears[state_of_choice_index]:
                transition = env.intermediate_boards[state_of_choice_index]
                image_list.append(env.draw(alternate_board=transition))
            image_list.append(env.draw())
        if print_steps:
            print('\rStep {}'.format(env.tetrominos_spawned), end='')
    if print_steps:
        print()

    return env.tetrominos_spawned

def simulate_without_pool(env, decision_maker, print_steps, log_images):
    """Without multiprocessing, simulates a game of Tetris with lookahead.

    Parameters
    ----------
    env : (Tetris)
        A Tetris environment to simulate a game within.
    decision_maker : (function handle)
        A function to determine which state to pursue. Must take in
        two inputs:
            boards (numpy.ndarray)
            line_clears (numpy.ndarray)
        The function then returns the index corresponding to the
        desired state to pursue.

    Returns
    -------
    env.tetrominos_spawned : (int)
        The number of tetrominos spawned before the game ended,
        equivalent to the number of steps taken.

    """

    env.reset()
    if log_images:
        image_list = [env.draw()]
    while not env.game_over:
        potential_boards, _ = env.generate_future_boards(None)

        potential_next_boards = []
        potential_next_clears = []
        initial_board_indexes = []
        for n, board in enumerate(potential_boards):
            next_boards, next_line_clears = env.generate_future_boards(board)
            potential_next_boards.append(next_boards)
            potential_next_clears.append(next_line_clears)
            initial_board_indexes.extend([n]*len(next_boards))

        potential_next_boards = np.concatenate(potential_next_boards)
        potential_next_clears = np.concatenate(potential_next_clears)

        next_board_of_choice_index = decision_maker(
                                            potential_next_boards,
                                            potential_next_clears)
        board_of_choice_index = initial_board_indexes[
                                            next_board_of_choice_index]
        env.step(potential_boards[board_of_choice_index])
        if log_images:
            if potential_next_clears[board_of_choice_index]:
                transition = env.intermediate_boards[board_of_choice_index]
                image_list.append(env.draw(alternate_board=transition))
            image_list.append(env.draw())
        if print_steps:
            print('\rStep {}'.format(env.tetrominos_spawned), end='')
    if print_steps:
        print()

    return env.tetrominos_spawned

#EOF