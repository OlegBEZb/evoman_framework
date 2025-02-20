import sys
import os
from utils import dict2str

# from evoman.environment import Environment
sys.path.insert(0, '/Users/Oleg_Litvinov1/Documents/Code/evoman_framework/evoman')
from environment import Environment
from controllers import PlayerController
from optimizers import EvolutionaryAlgorithm
from deap.tools.mutation import mutGaussian
from deap.tools.crossover import *
from custom_crossover import cx4ParentsCustomUniform, cxMultiParentUniform
from selection import selBest, selRandom, selTournament, selProportional

if __name__ == "__main__":
    try:
        LAUNCH_NUM = int(sys.argv[1])
        ENEMY_NUMBER = int(sys.argv[2])
    except:
        LAUNCH_NUM = 1
        ENEMY_NUMBER = 1

MATING_NUM = 3
POPULATION_SIZE = 20
PATIENCE = 17

DOOMSDAY_POPULATION_RATIO = 0.3
DOOMSDAY_REPLACE_WITH_RANDOM_PROB = 0.75

DEAP_CROSSOVER_METHOD = cx4ParentsCustomUniform
DEAP_CROSSOVER_KWARGS = {"indpb": 0.6}

DEAP_MUTATION_OPERATOR = mutGaussian
DEAP_MUTATION_KWARGS = {"mu": 0, "sigma": 1, "indpb": 0.8}

TOURNAMENT_METHOD = selProportional
TOURNAMENT_KWARGS = {'k': 4}

SURVIVOR_POOL = 'all'
SURVIVOR_SELECTION_METHOD = selBest

experiment_name = f"""experiments/enemy{ENEMY_NUMBER}_tournament{TOURNAMENT_METHOD.__name__}{dict2str(TOURNAMENT_KWARGS)}_mating{MATING_NUM}_pop{POPULATION_SIZE}_patience{PATIENCE}_DPR{DOOMSDAY_POPULATION_RATIO}_DRWRP{DOOMSDAY_REPLACE_WITH_RANDOM_PROB}_mutGaus_mu0sigma1prob{DEAP_MUTATION_KWARGS['indpb']}_{DEAP_CROSSOVER_METHOD.__name__}{dict2str(DEAP_CROSSOVER_KWARGS)}_LAUNCH_{LAUNCH_NUM}"""

print('experiment_name', experiment_name)
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

hidden_layer_sizes = [10, 5]
player_controller = PlayerController(hidden_layer_sizes)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[ENEMY_NUMBER],  # array with 1 to 8 items, values from 1 to 8
                  playermode="ai",
                  player_controller=player_controller,
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visualmode="no",  # requires environment adjustment
                  multiplemode='no',
                  randomini='yes')

ea = EvolutionaryAlgorithm(env=env,
                           experiment_name=experiment_name,
                           weight_amplitude=1,
                           mating_num=MATING_NUM,
                           population_size=POPULATION_SIZE,
                           patience=PATIENCE,
                           tournament_method=TOURNAMENT_METHOD,
                           tournament_kwargs=TOURNAMENT_KWARGS,
                           doomsday_population_ratio=DOOMSDAY_POPULATION_RATIO,
                           doomsday_replace_with_random_prob=DOOMSDAY_REPLACE_WITH_RANDOM_PROB,
                           deap_mutation_operator=DEAP_MUTATION_OPERATOR,
                           deap_mutation_kwargs=DEAP_MUTATION_KWARGS,
                           deap_crossover_method=DEAP_CROSSOVER_METHOD,
                           deap_crossover_kwargs=DEAP_CROSSOVER_KWARGS,
                           survivor_pool=SURVIVOR_POOL,  # can be all or offspring
                           survivor_selection_method=SURVIVOR_SELECTION_METHOD,
                           )
ea.train(generations=10)

env = Environment(experiment_name=experiment_name,
                  enemies=[1, 2, 3, 4, 5, 6, 7, 8],  # array with 1 to 8 items, values from 1 to 8
                  playermode="ai",
                  player_controller=player_controller,
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visualmode="yes",  # requires environment adjustment
                  multiplemode='yes',
                  randomini='yes')
ea.env = env
ea.test(n_times=5)
