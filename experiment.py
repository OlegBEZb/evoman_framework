import sys, os

# from evoman.environment import Environment
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import PlayerController
from optimizers import EvolutionaryAlgorithm
from deap.tools.mutation import mutGaussian
from deap.tools.crossover import cxOnePoint, cxUniform

if __name__ == "__main__":
    try:
        LAUNCH_NUM = int(sys.argv[1])
        ENEMY_NUMBER = int(sys.argv[2])
    except:
        LAUNCH_NUM = 1
        ENEMY_NUMBER = 1


MATING_NUM = 4
POPULATION_SIZE = 70
PATIENCE = 2
DOOMSDAY_POPULATION_RATIO = 0.3
DOOMSDAY_REPLACE_WITH_RANDOM_PROB = 0.9

experiment_name = f"experiments/enemy{ENEMY_NUMBER}_mating{MATING_NUM}_pop{POPULATION_SIZE}_patience{PATIENCE}_DPR{DOOMSDAY_POPULATION_RATIO}_DRWRP{DOOMSDAY_REPLACE_WITH_RANDOM_PROB}_mutGaus_03_cxOnePoint_{LAUNCH_NUM}"
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
                  visualmode="no", )

ea = EvolutionaryAlgorithm(env=env,
                           experiment_name=experiment_name,
                           weight_amplitude=1,
                           mating_num=MATING_NUM,
                           population_size=POPULATION_SIZE,
                           patience=PATIENCE,
                           doomsday_population_ratio=DOOMSDAY_POPULATION_RATIO,
                           doomsday_replace_with_random_prob=DOOMSDAY_REPLACE_WITH_RANDOM_PROB,
                           deap_mutation_operator=mutGaussian,
                           deap_mutation_kwargs={"mu": 0, "sigma": 1, "indpb": 0.3},
                           deap_crossover_method=cxOnePoint,
                           deap_crossover_kwargs={})
ea.train(generations=1)

ea.test()
