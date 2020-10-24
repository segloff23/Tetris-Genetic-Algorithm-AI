
import numpy as np

from collections import namedtuple

class Individual():
    """An indiviudal with certain genes (model_values) which determine choices.

    Methods
    -------
    initialize_model_values(None) --> (None)
        Runs the model's initializer on all parameters.
    clip_model_values(None) --> (None)
        Clips the model's values acoording to the given settings.
    mutate(int) --> (None):
        Mutates the genes of the indiviudal with given probability.
    """

    def __init__(self, model):
        """An indiviudal with certain genes to determine a value of a vector.

        Parameters
        ----------
        N : (int)
            The number of ratings produced by heuristics.

        Returns
        -------
        None.

        """

        self.model = model

        self.model_values = []
        self.initialize_model_values()
        self.clip_model_values()

    def initialize_model_values(self):
        """Runs the model's initializer on all parameters.
        """
        for settings in self.model:
            self.model_values += [settings.init(settings.size).astype(settings.dtype)]

    def clip_model_values(self):
        """Clips the model's values acoording to the given settings.
        """
        for value, settings in zip(self.model_values, self.model):
            if settings.limit is not None:
                value[:] = np.clip(value, *settings.limit)

    def mutate(self, mut_prob):
        """Mutates the genes of the indiviudal with given probability.
        Parameters
        ----------
        mut_prob : (int)
            Probability of a gene being mutated.

        Returns
        -------
        None.

        """
        for value, settings in zip(self.model_values, self.model):

            mut_selection = np.random.choice([0, 1], size=settings.size,
                                             p=[1-mut_prob, mut_prob])
            mutated_value = (value *
                        settings.mutator(settings.size).astype(settings.dtype))
            value[:] = np.where(mut_selection, mutated_value, value)

        self.clip_model_values()

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
    def __init__(self, env, heuristics, generation_size, base_model, evaluator,
                 model_file_name):
        """Environment within which evolution takes place to optimize fitness.

        Parameters
        ----------
        env : (tetris.Tetris)
            Tetris environment in which fitness will be measured.
        heuristics : (heuristics.Heuristics)
            Ratings environment to feed to individuls.
        generation_size : (int)
            Number of indiviudals within the generation.
        base_model : (MODEL.model)
            Named tuple containing chromosones and their settings.
        evaluator : (function handle)
            Function used to evaluate boards using model values and
            the boards' ratings.

        Returns
        -------
        None.

        """

        self.env = env
        self.heuristics = heuristics
        self.generation_size = generation_size
        self.base_model = base_model
        self.evaluator = evaluator
        self.model_file_name = model_file_name

        self.N = heuristics.determine_values([env.current_board],
                                             [0]).shape[0]

        self.model = self.update_base_model()
        self.generation = [Individual(self.model)
                           for _ in range(generation_size)]

        self.generation_number = 0
        self.fitness_scores = None
        self.num_elites = None

    def update_base_model(self):

        new_settings = {}
        for field, settings in zip(self.base_model._fields, self.base_model):
            true_size = []
            for d in settings.size:
                if d is None:
                    true_size.append(self.N)
                else:
                    true_size.append(d)
            true_size = tuple(true_size)
            new_settings[field] = settings._replace(size=true_size)

        return self.base_model._replace(**new_settings)

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
            decision_maker = lambda B, C : (
                self.evaluator(*individual.model_values,
                               self.heuristics.determine_values(B, C)))

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

        child = Individual(self.model)

        # TODO: Experiment with applying two-point crossover
        # to all chromosones (if there are mutliple) at once,
        # so their relationship is preserved

        for v_child, v_husband, v_wife in zip(child.model_values,
                                              husband.model_values,
                                              wife.model_values):
            slices = np.random.choice(self.N, size=(2,), replace=False)
            slices.sort()
            a, b = slices
            v_child[0:a] = v_husband[0:a]
            v_child[a:b] = v_wife[a:b]
            v_child[b::] = v_husband[b::]

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

        for child in children:
            child.mutate(mutation_probability)

        self.generation = [self.generation[k] for k in elite_set] + children
        self.generation_size = generation_size
        self.fitness_scores = fitness_scores
        self.num_elites = num_elites
        self.generation_number += 1

        return mean_population_score, mean_elite_score

def generate_model(model_names, model_sizes, model_limits,
                   model_initializers, model_dtypes, model_mutators):
    """Combines various model parameters into the MODEL named tuple class.

    Parameters
    ----------
    model_names : (list of strings)
        The names of each chromosone in the model.
    model_sizes : (list of tuples of ints or None)
        The dimensions of each chromosone. Use None to denote the dimension
        should be filled with that of the ratings function from heuristics.
    model_limits : (list of None or tuples of any num type or None)
        Limits the given chromosones should be clipped to. Use just None
        to denote no clipping, and (None, a) or (a, None) to clip either
        only above or only below.
    model_initializers : (function handles)
        Functions to initialize the model with. Must take in a tuple
        representing the size of the chromosone, identical to model_sizes
    model_dtypes : (strings)
        Name of data type to be used for that parameter,
        i.e., 'int32' or 'float64.'
    model_mutators : (function handles)
        Functions to mutate the models with via multiplication,
        i.e., new_model = old_model * mutation.

    Returns
    -------
    MODEL.model
        The MODEL object used to determine how individuals behave.

    """
    global MODEL # Necessary for pickling
    global SETTINGS # Necessary for pickling

    MODEL = namedtuple('MODEL', model_names)
    SETTINGS = namedtuple('SETTINGS', ['size', 'limit', 'init', 'dtype', 'mutator'])
    model_input = []
    for name, size, limit, init, dtype, mutator in zip(model_names, model_sizes,
                                                model_limits, model_initializers,
                                                model_dtypes, model_mutators):
        model_input += [SETTINGS(size, limit, init, dtype, mutator)]

    return MODEL(*model_input)

#EOF