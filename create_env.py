import sys, os
# print(os.environ.get('EVOMAN_PATH'))
sys.path.insert(0, '/Users/Oleg_Litvinov1/Documents/Code/evoman_framework/evoman')
# sys.path.insert(0, os.environ.get('EVOMAN_PATH'))

from environment import Environment
from controllers import PlayerController

def create_env(experiment_name, enemy_number, multiplemode='yes'):
    hidden_layer_sizes = [10, 5]
    player_controller = PlayerController(hidden_layer_sizes)
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=enemy_number,  # array with 1 to 8 items, values from 1 to 8
                      playermode="ai",
                      player_controller=player_controller,
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visualmode="no",
                      multiplemode=multiplemode,
                      randomini='yes')
    return env