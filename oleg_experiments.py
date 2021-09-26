import sys, os
# from evoman.environment import Environment
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import PlayerController
from optimizers import EvolutionaryAlgorithm

experiment_name = 'mating_num4_mut01_pop100'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


hidden_layer_sizes = [10]
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
                           population_size=100,
                           mutation=0.1,
                           patience=15)
ea.train(generations=30)

ea.test()
