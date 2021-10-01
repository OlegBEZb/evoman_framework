from deap.tools.crossover import cxUniform

def crossover_4_default(parent1, parent2, parent3, parent4, indpb):
    ch1 = cxUniform(parent1, parent2, indpb)[0]
    ch2 = cxUniform(parent3, parent4, indpb)[0]
    ch1_1 = cxUniform(ch1, parent3, indpb)[0]
    ch2_2 = cxUniform(ch2, parent2, indpb)[0]
    return ch1_1, ch2_2
