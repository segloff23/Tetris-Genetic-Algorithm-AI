import numpy as np

class Value_Function():

    def __init__(self, heuristics, N):

        self.heuristics = heuristics
        self.N = N

        self.weights = np.array(np.random.normal(size=(self.N,1),
                                                 scale=50),
                                dtype='float32')
        self.weights /= np.max(np.abs(self.weights))

        self.parameters = [self.weights]

    def evaluate_states(self, boards, clears):

        raw_values = self.heuristics.determine_values(boards, clears)
        evaluation = np.sum(self.weights * raw_values, axis=0)

        return np.argmax(evaluation)

    def mutate(self, mutation):
        self.weights *= ((mutation[0] *
                          np.array(np.random.normal(size=(self.N,1),
                                                    scale=2),
                                   dtype='float32'))
                         + ((1 - mutation[0]) * np.ones((self.N,1),
                                                        dtype='float32')))

class Ecosystem():

    def __init__(self, environment, simulator, heuristics, generation_size):

        self.environment = environment
        self.simulator = simulator
        self.heuristics = heuristics
        self.generation_size = generation_size

        self.N = heuristics.determine_values([environment.current_board],
                                             [0]).shape[0]
        self.generation = [Value_Function(heuristics, self.N)
                           for _ in range(generation_size)]
        self.param_count = len(self.generation[0].parameters)
        
        self.fitness_scores = None
        self.num_elites = None

    def evaluate_population(self, pool, number_of_games):

        all_time_best = 0
        rewards_achieved = []
        for n, individual in enumerate(self.generation):

            reward = 0
            for game in range(number_of_games):
                reward += self.simulator(pool, self.environment,
                                         individual.evaluate_states)

            if reward > all_time_best:
                all_time_best = reward

            rewards_achieved.append(reward / number_of_games)
            
            s = '\r{:d} / {:<3d} Individuals Tested [Best Score: {:d}]'
            print(s.format(n+1, self.generation_size, all_time_best), end='')
        print()
        return rewards_achieved

    def mate(self, husband, wife):

        child = Value_Function(self.heuristics, self.N)

        # TODO: Experiment with applying two-point crossover
        # to all chromosones at once, so their relationship is preserved
        for p_child, p_husband, p_wife in zip(child.parameters,
                                              husband.parameters,
                                              wife.parameters):

            a, b = np.random.choice(self.N, size=(2,), replace=False)
            dx, dy = np.random.randint(-10, 11, size=(2,))

            p_child[0:a] = dx * p_husband[0:a]
            p_child[a:b] = dy * p_wife[a:b]
            p_child[b::] = dx * p_husband[b::]

        return child

    def evolve(self,
               pool, number_of_games, generation_size, mutation_probability):

        fitness_scores = np.array(self.evaluate_population(pool,
                                                           number_of_games))
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
            for param in child.parameters:
                param /= np.max(np.abs(param))

        self.generation = [self.generation[k] for k in elite_set] + children
        self.generation_size = generation_size
        self.fitness_scores = fitness_scores
        self.num_elites = num_elites

        return mean_population_score, mean_elite_score



