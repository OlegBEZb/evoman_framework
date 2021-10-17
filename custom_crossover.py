from deap.tools.crossover import cxUniform
import numpy as np
from copy import deepcopy


def cx4ParentsCustomUniform(parents, **kwargs):
    parent1, parent2, parent3, parent4 = parents
    ch1 = cxUniform(parent1, parent2, **kwargs)[0]
    ch2 = cxUniform(parent3, parent4, **kwargs)[0]
    ch1_1 = cxUniform(ch1, parent3, **kwargs)[0]
    ch2_2 = cxUniform(ch2, parent2, **kwargs)[0]
    return ch1_1, ch2_2


def cxMultiParentUniform(parents):
    """Executes a uniform crossover

    :param parents: list of individuals any size
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    parents_num = len(parents)
    individual_length = min([len(x) for x in parents])
    offspring = deepcopy(parents)
    for i in range(parents_num):  # vectorize for each children
        for gene_number in range(individual_length):  # for each gene
            selected_gene_parent = np.random.choice(a=range(parents_num))
            offspring[i][gene_number] = parents[selected_gene_parent][gene_number]

    return offspring
