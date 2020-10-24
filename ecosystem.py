import numpy as np

class Value_Function():
    """An indiviudal with certain genes to determine a value of a vector.

    Methods
    -------
    evaluate_states(numpy.ndarray, numpy.ndarray) --> (int)
        Determines the index of the optimal board within the given list.
    mutate(numpy.ndarray) --> (None):
        Mutates the genes of the indiviudal according given by the input.
    """

    def __init__(self, N):
        """An indiviudal with certain genes to determine a value of a vector.

        Parameters
        ----------
        N : (int)
            The number of ratings produced by heuristics.

        Returns
        -------
        None.

        """

        self.N = N

        self.weights = np.array(np.random.normal(size=(self.N,1),
                                                 scale=50),
                                dtype='float32')
        self.weights /= np.max(np.abs(self.weights))
        
        self.exponents = np.abs(np.array(np.random.normal(size=(self.N,1),
                                                 scale=2),
                                         dtype='float32'))
        self.exponents = np.clip(self.exponents, 0, 5)

        self.parameters = [self.weights, self.exponents]

    def evaluate_states(self, ratings):
        """Determines the index of the optimal board within the given list.

        Parameters
        ----------
        ratings : (numpy.ndarray)
            Ratings vector for each board.

        Returns
        -------
        (int)
            The index of the optimal board.

        """

        evaluation = np.sum(self.weights
                            * (np.power(ratings, self.exponents)), axis=0)

        return np.argmax(evaluation)

    def mutate(self, mutation):
        """Mutates the genes of the indiviudal according given by the input.
        Parameters
        ----------
        mutation : (numpy.ndarray)
            Array describing which genes of which parameters to update.

        Returns
        -------
        None.

        """
        
        for k, param in enumerate(self.parameters):
            if k == 0:
                param *= ((mutation[0] *
                          np.array(np.random.normal(size=(self.N,1),
                                                    scale=2),
                                   dtype='float32'))
                         + ((1 - mutation[0]) * np.ones((self.N,1),
                                                        dtype='float32')))
            elif k == 1:
                param *= ((mutation[1] *
                            np.abs(np.array(np.random.normal(size=(self.N,1),
                                                    scale=2),
                                   dtype='float32')))
                           + ((1 - mutation[1]) * np.ones((self.N,1),
                                                        dtype='float32')))
                param[:] = np.clip(param, 0, 5)

class Ecosystem():
    """Environment within which evolution takes place to optimize fitness.

    Methods
    -------
    evaluate_population(int, Pool.pool) --> (list)
        Determines the fitness of each indiviudal in the current generation.
    mate(Value_Function, Value_Function) --> Value Function
        Mixes two parents with 2 index slicing.
    evolve(int, int, int, Pool.pool) --> (float, float)
        Evolves the current generation into a new one using their fitness
        within the environment.
    """
    def __init__(self, env, heuristics, generation_size):
        """Environment within which evolution takes place to optimize fitness.

        Parameters
        ----------
        env : (tetris.Tetris)
            Tetris environment in which fitness will be measured.
        heuristics : (heuristics.Heuristics)
            Ratings environment to feed to individuls.
        generation_size : (int)
            Number of indiviudals within the generation.

        Returns
        -------
        None.

        """

        self.env = env
        self.heuristics = heuristics
        self.generation_size = generation_size
        self.generation_number = 0

        self.N = heuristics.determine_values([env.current_board],
                                             [0]).shape[0]
        self.generation = [Value_Function(self.N)
                           for _ in range(generation_size)]
        self.param_count = len(self.generation[0].parameters)

        self.fitness_scores = None
        self.num_elites = None

    def evaluate_population(self, number_of_games, pool=None):
        """Determine the fitness of the current generation.

        Parameters
        ----------
        number_of_games : (int)
            Number of games to average fitness over.
        pool : (Pool.pool) or (None), optional
            Pool to perform multiprocessing over. The default is None.

        Returns
        -------
        rewards_achieved : (list)
            The success/fitness of each indiviudal.

        """

        all_time_best = 0
        rewards_achieved = []
        for n, individual in enumerate(self.generation):
            decision_maker = lambda x,y : (individual.evaluate_states(
                                    self.heuristics.determine_values(x, y)))
            
            reward = 0
            for game in range(number_of_games):
                reward += self.env.simulate(decision_maker, pool=pool)

            if reward > all_time_best:
                all_time_best = reward

            rewards_achieved.append(reward / number_of_games)

            s = '\r{:d} / {:<3d} Individuals Tested [Best Score: {:d}]'
            print(s.format(n+1, self.generation_size, all_time_best), end='')
        print()

        return rewards_achieved

    def mate(self, husband, wife):
        """Mixes two parents with 2 index slicing.

        Parameters
        ----------
        husband : Value_Function
            One of the parents to draw genes from.
        wife : Value_Function
            One of the parents to draw genes from.

        Returns
        -------
        child : Value_Function
            The new individual with genes from both parents.

        """

        child = Value_Function(self.N)

        # TODO: Experiment with applying two-point crossover
        # to all chromosones (if there are mutliple) at once,
        # so their relationship is preserved
        k = 0
        for p_child, p_husband, p_wife in zip(child.parameters,
                                              husband.parameters,
                                              wife.parameters):

            a, b = np.random.choice(self.N, size=(2,), replace=False)
            if k == 0:
                dx, dy = np.random.randint(-10, 11, size=(2,))
                k += 1
            else:
                dx, dy = 1, 1

            p_child[0:a] = dx * p_husband[0:a]
            p_child[a:b] = dy * p_wife[a:b]
            p_child[b::] = dx * p_husband[b::]

        return child

    def evolve(self, number_of_games, generation_size,
               mutation_probability, pool=None):
        """Evolves the current generation into a new one using their fitness.

        Parameters
        ----------
        number_of_games : (int)
            The number of games to average fitness over.
        generation_size : (int)
            The size of the new generation.
        mutation_probability : (float)
            Probability a single gene is mutated.
        pool : (Pool.pool) or (None), optional
            Pool to perform multiprocessing over. The default is None.

        Returns
        -------
        mean_population_score : (float)
            The mean fitness of the generation evolved.
        mean_elite_score : (float)
            The mean fitness of the best performers of the generation evolved.

        """

        fitness_scores = np.array(self.evaluate_population(number_of_games,
                                                           pool=pool))
        mean_population_score = np.sum(fitness_scores) / self.generation_size

        elite_set = np.zeros((0,))
        percentile = 80
        while elite_set.shape[0] < 2:
            elite_score = np.percentile(fitness_scores, percentile)
            elite_set = np.where(fitness_scores > elite_score)[0]
            percentile -= 5
        mean_elite_score = np.mean(fitness_scores[elite_set])

        num_elites = len(elite_set)
        reproduction_probability = [int(score >= elite_score)
                                    for score in fitness_scores]
        rescale = sum(reproduction_probability)
        for k in range(self.generation_size):
            reproduction_probability[k] /= rescale

        parents_list = np.random.choice(self.generation,
                                        size=(generation_size-num_elites,2),
                                        p=reproduction_probability)
        children = [self.mate(*parents) for parents in parents_list]

        mutation_occurences = np.random.choice([0, 1],
                                               size=(generation_size,
                                                     self.param_count,
                                                     self.N, 1),
                                               p=[1-mutation_probability,
                                                  mutation_probability])
        for child, mutation in zip(children, mutation_occurences):
            child.mutate(mutation)

        for child in children:
            scale = np.max(np.abs(child.parameters[0]))
            if scale != 0:
                child.parameters[0] /= scale
            else:
                child.parameters[0][:] = np.random.normal(size=(self.N,1), scale=50)

        self.generation = [self.generation[k] for k in elite_set] + children
        self.generation_size = generation_size
        self.fitness_scores = fitness_scores
        self.num_elites = num_elites
        self.generation_number += 1

        return mean_population_score, mean_elite_score



