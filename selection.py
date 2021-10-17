import numpy as np
from operator import itemgetter
from sklearn.preprocessing import MinMaxScaler
from utils import norm


def selRandom(population, k):
    """Select *k* individuals at random from the input *individuals* with
    replacement. The list returned contains references to the input
    *individuals*.
    :param population: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    This function uses the :func:`~random.choice` function from the
    python base :mod:`random` module.
    """
    if k >= len(population):
        replace = True
    else:
        replace = False
    rows = np.random.choice(population.shape[0], k, replace=replace)
    individuals = population[rows]
    return individuals


def selBest(population, population_fitness, k):
    """Select the *k* best individuals among the input *individuals*. The
    list returned contains references to the input *individuals*.
    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list containing the k best individuals.
    """
    return [x for x, _ in sorted(zip(population, population_fitness), key=lambda pair: pair[1])][:k]


def selTournament(population, population_fitness, k, tournsize):
    # ATTENTION: this func is slow
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.
    :param population: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :returns: A list of selected individuals.
    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    chosen = []
    for i in range(k):
        indices = np.random.choice(range(len(population_fitness)), size=tournsize, replace=False)
        max_index = np.argmax(itemgetter(*indices)(population_fitness))
        chosen.append(population[indices[max_index]])
    return chosen


def selProportional(population, population_fitness, k):
    # avoiding negative probabilities, as fitness is ranges from negative numbers
    population_fitness = np.array(list(map(lambda y: norm(y, population_fitness), population_fitness)))
    probs = population_fitness / population_fitness.sum()
    if k >= len(population):
        replace = True
    else:
        replace = False
    rows = np.random.choice(population.shape[0], size=k, p=probs.ravel(), replace=replace)
    return population[rows]
