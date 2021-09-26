import sys, os
# from evoman.environment import Environment
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import PlayerController
from optimizers import EvolutionaryAlgorithm
from deap.tools.mutation import mutGaussian
from deap.tools.crossover import cxOnePoint, cxUniform


experiment_name = 'mating4_pop50_mutGaus_03_cxUniform05'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


hidden_layer_sizes = [10, 5]
player_controller = PlayerController(hidden_layer_sizes)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller,
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visualmode="no",)

ea = EvolutionaryAlgorithm(env=env,
                           experiment_name=experiment_name,
                           weight_amplitude=1,
                           mating_num=4,
                           population_size=50,
                           patience=5,
                           deap_mutation_operator=mutGaussian,
                           deap_mutation_kwargs={"mu": 0, "sigma": 1, "indpb": 0.3},
                           deap_crossover_method=cxUniform,
                           deap_crossover_kwargs={'indpb': 0.5})
ea.train(generations=20)

ea.test()
