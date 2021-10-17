from deap.tools.crossover import cxUniform


def cx4ParentsCustomUniform(parent1, parent2, parent3, parent4, **kwargs):
    ch1 = cxUniform(parent1, parent2, **kwargs)[0]
    ch2 = cxUniform(parent3, parent4, **kwargs)[0]
    ch1_1 = cxUniform(ch1, parent3, **kwargs)[0]
    ch2_2 = cxUniform(ch2, parent2, **kwargs)[0]
    return ch1_1, ch2_2
