from hyperopt import fmin, hp, tpe
from hyperopt import SparkTrials, STATUS_OK
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
from hyperopt import plotting
import sys, os, pickle

os.environ['JAVA_HOME'] = "C:\Program Files\Java\jdk-11.0.11"
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# from evoman.environment import Environment
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import PlayerController
from optimizers import EvolutionaryAlgorithm
import numpy as np

from deap.tools.mutation import mutGaussian
from deap.tools.crossover import *
from selection import selBest, selRandom, selTournament, selProportional

if __name__ == "__main__":
    try:
        LAUNCH_NUM = int(sys.argv[1])
        ENEMY_NUMBER = int(sys.argv[2])
    except:
        LAUNCH_NUM = 1
        ENEMY_NUMBER = [2, 3, 5]


def train(params):
    """
  This is our main training function which we pass to Hyperopt.
  It takes in hyperparameter settings, fits a model based on those settings,
  evaluates the model, and returns the loss.

  :param params: map specifying the hyperparameter settings to test
  :return: loss for the fitted model
  """
    params = params['type']

    mating_num = params['mating_num']
    population_size = params['population_size']
    patience = params['patience']

    doomsday_population_ratio = params['doomsday_population_ratio']
    doomsday_replace_with_random_prob = params['doomsday_replace_with_random_prob']

    deap_crossover_method = params['deap_crossover_method']
    deap_crossover_kwargs = params.get('deap_crossover_kwargs', {})

    deap_mutation_operator = params['deap_mutation_operator']
    deap_mutation_kwargs = params.get('deap_mutation_kwargs', {})

    tournament_method = params['tournament_method']
    tournament_kwargs = params.get('tournament_kwargs', {})

    experiment_name = f"experiments/enemy{ENEMY_NUMBER}_tournament{tournament_method.__name__}_mating{mating_num}_pop{population_size}_patience{patience}_DPR{doomsday_population_ratio}_DRWRP{doomsday_replace_with_random_prob}_mutGaus_mu0sigma1prob{deap_mutation_kwargs['indpb']}_{deap_crossover_method.__name__}_hyperopt"
    print('experiment_name', experiment_name)
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    hidden_layer_sizes = [10, 5]
    player_controller = PlayerController(hidden_layer_sizes)

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=ENEMY_NUMBER,  # array with 1 to 8 items, values from 1 to 8
                      playermode="ai",
                      player_controller=player_controller,
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visualmode="no",
                      multiplemode='yes',
                      randomini='yes')

    ea = EvolutionaryAlgorithm(env=env,
                               experiment_name=experiment_name,
                               weight_amplitude=1,
                               mating_num=mating_num,
                               population_size=population_size,
                               patience=patience,
                               tournament_method=tournament_method,
                               tournament_kwargs=tournament_kwargs,
                               doomsday_population_ratio=doomsday_population_ratio,
                               doomsday_replace_with_random_prob=doomsday_replace_with_random_prob,
                               deap_mutation_operator=deap_mutation_operator,
                               deap_mutation_kwargs=deap_mutation_kwargs,
                               deap_crossover_method=deap_crossover_method,
                               deap_crossover_kwargs=deap_crossover_kwargs)
    ea.train(generations=20)
    fitness_list = ea.test(n_times=5)

    return {'loss': -np.mean(fitness_list), 'status': STATUS_OK}


# Next, define a search space for Hyperopt.
search_space = {
    'type': hp.choice('type', [
        {
            "type": 1,
            "mating_num": ho_scope.int(hp.quniform("mating_num1", 1, 10, q=1)),
            "population_size": hp.choice("population_size1", [30, 100]),
            "patience": ho_scope.int(hp.quniform("patience1", 1, 20, q=1)),

            "doomsday_population_ratio": hp.uniform("doomsday_population_ratio1", 0, 1),
            "doomsday_replace_with_random_prob": hp.uniform("doomsday_replace_with_random_prob1", 0, 1),

            "deap_crossover_method": hp.choice("deap_crossover_method1", [cxUniform]),
            "deap_crossover_kwargs": hp.uniform("deap_crossover_kwargs1", 0, 1),

            "deap_mutation_operator": hp.choice("deap_mutation_operator1", [mutGaussian]),
            "deap_mutation_kwargs": hp.choice("deap_mutation_kwargs1", [{"mu": 0, "sigma": 1, "indpb": 0.3},
                                                                        {"mu": 0, "sigma": 1, "indpb": 0.6},
                                                                        {"mu": 0, "sigma": 1, "indpb": 0.2},
                                                                        {"mu": 0, "sigma": 1, "indpb": 0.8}]),

            "tournament_method": hp.choice("tournament_method1", [selTournament]),
            "tournament_kwargs": hp.choice("tournament_kwargs1", [{"k": 2, "tournsize": 4},
                                                                  {"k": 2, "tournsize": 10}]),
        },
        {
            "type": 2,
            "mating_num": ho_scope.int(hp.quniform("mating_num", 1, 10, q=1)),
            "population_size": hp.choice("population_size2", [30, 100]),
            "patience": ho_scope.int(hp.quniform("patience", 1, 20, q=1)),
            "doomsday_population_ratio": hp.uniform("doomsday_population_ratio", 0, 1),
            "doomsday_replace_with_random_prob": hp.uniform("doomsday_replace_with_random_prob", 0, 1),
            "deap_crossover_method": hp.choice("deap_crossover_method", [cxUniform]),

            "deap_mutation_operator": hp.choice("deap_mutation_operator", [mutGaussian]),
            "deap_mutation_kwargs": hp.choice("deap_mutation_kwargs", [{"mu": 0, "sigma": 1, "indpb": 0.3},
                                                                       {"mu": 0, "sigma": 1, "indpb": 0.6},
                                                                       {"mu": 0, "sigma": 1, "indpb": 0.2},
                                                                       {"mu": 0, "sigma": 1, "indpb": 0.8}]),

            "tournament_method": hp.choice("tournament_method", [selProportional]),
            "tournament_kwargs": hp.choice("tournament_kwargs1", [{"k": 2}]),  # may be hardcoded?
        }
    ])}

# for i in range(5):
#     print(ho_sample(search_space))

# Select a search algorithm for Hyperopt to use.
algo = tpe.suggest  # Tree of Parzen Estimators, a Bayesian method

# # We can run Hyperopt locally (only on the driver machine)
# # by calling `fmin` without an explicit `trials` argument.
best_hyperparameters = fmin(
    fn=train,
    space=search_space,
    algo=algo,
    max_evals=2,
    timeout=25200)

# We can distribute tuning across our Spark cluster
# by calling `fmin` with a `SparkTrials` instance.
# spark_trials = SparkTrials()
# best_hyperparameters = fmin(
#   fn=train,
#   space=search_space,
#   algo=algo,
#   trials=spark_trials,
#   max_evals=32)

# for _ in range(32):
#     best_hyperparameters = fmin(
#         fn=train,
#         space=search_space,
#         algo=algo,
#         trials=spark_trials,
#         max_evals=len(spark_trials)+1)
#     plotting.main_plot_history(spark_trials)

print('best_hyperparameters', best_hyperparameters)
