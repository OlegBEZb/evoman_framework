import numpy as np
from operator import itemgetter


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
    rows = np.random.choice(population.shape[0], k, replace=False)
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
        indices = np.random.choice(range(len(population_fitness)), size=tournsize, replace=True)
        max_index = np.argmax(itemgetter(*indices)(population_fitness))
        chosen.append(population[indices[max_index]])
    return chosen
