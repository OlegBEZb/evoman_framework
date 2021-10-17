from hyperopt import fmin, hp, tpe
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample
from hyperopt import plotting
import sys, os

from create_env import create_env

from optimizers import EvolutionaryAlgorithm

from deap.tools.mutation import mutGaussian
from deap.tools.crossover import *
from custom_crossover import *
from selection import selBest, selRandom, selTournament, selProportional

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

if __name__ == "__main__":
    try:
        LAUNCH_NUM = int(sys.argv[1])
        ENEMY_NUMBER = int(sys.argv[2])
    except:
        LAUNCH_NUM = 1
        ENEMY_NUMBER = [2, 3, 5]

TUNING_HOURS = 3
N_GENERATIONS = 5

root = os.getcwd()


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
    if not isinstance(deap_crossover_kwargs, dict):
        deap_crossover_kwargs = {"indpb": deap_crossover_kwargs}

    deap_mutation_operator = params['deap_mutation_operator']
    deap_mutation_kwargs = params.get('deap_mutation_kwargs', {})

    tournament_method = params['tournament_method']
    tournament_kwargs = params.get('tournament_kwargs', {})

    survivor_pool = params['survivor_pool']
    survivor_selection_method = params['survivor_selection_method']

    enemy_number = params['enemy_number']

    experiment_name = f"experiments/enemy{enemy_number}_tournament{tournament_method.__name__}_mating{mating_num}_pop{population_size}_patience{patience}_DPR{doomsday_population_ratio}_DRWRP{doomsday_replace_with_random_prob}_mutGaus_mu0sigma1prob{deap_mutation_kwargs['indpb']}_{deap_crossover_method.__name__}_survival_{survivor_pool}_{survivor_selection_method.__name__}_generalist"
    print('experiment_name', experiment_name)
    experiment_name = os.path.join(root, experiment_name)
    print('path name', experiment_name)
    if not os.path.exists(experiment_name):
        print('creating folder')
        os.makedirs(experiment_name)

    env = create_env(experiment_name, enemy_number)

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
                               deap_crossover_kwargs=deap_crossover_kwargs,
                               survivor_pool=survivor_pool,
                               survivor_selection_method=survivor_selection_method,
                               )
    ea.train(generations=N_GENERATIONS)

    env = create_env(experiment_name, [1, 2, 3, 4, 5, 6, 7, 8])
    ea.env = env
    fitness_list = ea.test(n_times=5)

    # Send the current training result back to Tune
    tune.report(mean_accuracy=np.mean(fitness_list))

    # return {'loss': -np.mean(fitness_list), 'status': STATUS_OK}


# Next, define a search space for Hyperopt.
search_space = {
    'type': hp.choice('type', [
        # {
        #     "type": 'tournament',
        #     "mating_num": ho_scope.int(hp.quniform("mating_num1", 1, 10, q=1)),
        #     "population_size": hp.choice("population_size1", [30, 70]),
        #     "patience": ho_scope.int(hp.quniform("patience1", 1, N_GENERATIONS, q=1)),
        #
        #     "doomsday_population_ratio": hp.uniform("doomsday_population_ratio1", 0, 1),
        #     "doomsday_replace_with_random_prob": hp.uniform("doomsday_replace_with_random_prob1", 0, 1),
        #
        #     "deap_crossover_method": hp.choice("deap_crossover_method1", [cxUniform]),  # tournament_kwargs['k'] should be correspondent
        #     "deap_crossover_kwargs": hp.uniform("deap_crossover_kwargs1", 0, 1),
        #
        #     "deap_mutation_operator": hp.choice("deap_mutation_operator1", [mutGaussian]),
        #     "deap_mutation_kwargs": hp.choice("deap_mutation_kwargs1", [{"mu": 0, "sigma": 1, "indpb": 0.3},
        #                                                                 {"mu": 0, "sigma": 1, "indpb": 0.6},
        #                                                                 {"mu": 0, "sigma": 1, "indpb": 0.2},
        #                                                                 {"mu": 0, "sigma": 1, "indpb": 0.8}]),
        #
        #     "tournament_method": hp.choice("tournament_method1", [selTournament]),
        #     "tournament_kwargs": hp.choice("tournament_kwargs1", [{"k": 2, "tournsize": 4},
        #                                                           {"k": 2, "tournsize": 10}]),
        #     "enemy_number": hp.choice("enemy_number1", [[1, 4, 7], [2, 4, 8], [1, 5, 6], [3, 5, 8]]),
        #     "survivor_pool": hp.choice("survivor_pool1", ['all', 'offspring']),
        #     "survivor_selection_method": hp.choice("survivor_selection_method1", [selProportional, selBest]),
        # },
        # {
        #     "type": 'proportional',
        #     "mating_num": ho_scope.int(hp.quniform("mating_num", 1, 10, q=1)),
        #     "population_size": hp.choice("population_size2", [30, 70]),
        #     "patience": ho_scope.int(hp.quniform("patience", 1, N_GENERATIONS, q=1)),
        #     "doomsday_population_ratio": hp.uniform("doomsday_population_ratio", 0, 1),
        #     "doomsday_replace_with_random_prob": hp.uniform("doomsday_replace_with_random_prob", 0, 1),
        #
        #     "deap_crossover_method": hp.choice("deap_crossover_method", [cxUniform]),
        #     "deap_crossover_kwargs": hp.uniform("deap_crossover_kwargs", 0, 1),
        #
        #     "deap_mutation_operator": hp.choice("deap_mutation_operator", [mutGaussian]),
        #     "deap_mutation_kwargs": hp.choice("deap_mutation_kwargs", [{"mu": 0, "sigma": 1, "indpb": 0.3},
        #                                                                {"mu": 0, "sigma": 1, "indpb": 0.6},
        #                                                                {"mu": 0, "sigma": 1, "indpb": 0.2},
        #                                                                {"mu": 0, "sigma": 1, "indpb": 0.8}]),
        #
        #     "tournament_method": hp.choice("tournament_method", [selProportional]),
        #     "tournament_kwargs": hp.choice("tournament_kwargs", [{"k": 2}]),  # may be hardcoded?
        #     "enemy_number": hp.choice("enemy_number", [[1, 2, 3], [6, 7, 8], [1, 4, 8], [3, 7, 8]]),
        #     "survivor_pool": hp.choice("survivor_pool2", ['all', 'offspring']),
        #     "survivor_selection_method": hp.choice("survivor_selection_method2", [selProportional, selBest]),
        # },
{
            "type": 'tournament_4_parents',
            "mating_num": ho_scope.int(hp.quniform("mating_num3", 1, 10, q=1)),
            "population_size": hp.choice("population_size3", [30, 70]),
            "patience": ho_scope.int(hp.quniform("patience3", 1, N_GENERATIONS, q=1)),

            "doomsday_population_ratio": hp.uniform("doomsday_population_ratio3", 0, 1),
            "doomsday_replace_with_random_prob": hp.uniform("doomsday_replace_with_random_prob3", 0, 1),

            "deap_crossover_method": hp.choice("deap_crossover_method3", [cx4ParentsCustomUniform]),
            "deap_crossover_kwargs": hp.uniform("deap_crossover_kwargs3", 0, 1),

            "deap_mutation_operator": hp.choice("deap_mutation_operator3", [mutGaussian]),
            "deap_mutation_kwargs": hp.choice("deap_mutation_kwargs3", [{"mu": 0, "sigma": 1, "indpb": 0.3},
                                                                        {"mu": 0, "sigma": 1, "indpb": 0.6},
                                                                        {"mu": 0, "sigma": 1, "indpb": 0.2},
                                                                        {"mu": 0, "sigma": 1, "indpb": 0.8}]),

            "tournament_method": hp.choice("tournament_method3", [selTournament]),
            "tournament_kwargs": hp.choice("tournament_kwargs3", [{"k": 4, "tournsize": 8},
                                                                  {"k": 4, "tournsize": 5}]),
            "enemy_number": hp.choice("enemy_number3", [[1, 4, 7], [2, 4, 8], [1, 2], [5, 8]]),
            "survivor_pool": hp.choice("survivor_pool3", ['all', 'offspring']),
            "survivor_selection_method": hp.choice("survivor_selection_method3", [selProportional, selBest]),
        },
{
            "type": 'tournament_multi_parents',
            "mating_num": ho_scope.int(hp.quniform("mating_num1", 1, 10, q=1)),
            "population_size": hp.choice("population_size1", [30, 70]),
            "patience": ho_scope.int(hp.quniform("patience1", 1, N_GENERATIONS, q=1)),

            "doomsday_population_ratio": hp.uniform("doomsday_population_ratio1", 0, 1),
            "doomsday_replace_with_random_prob": hp.uniform("doomsday_replace_with_random_prob1", 0, 1),

            "deap_crossover_method": hp.choice("deap_crossover_method1", [cxMultiParentUniform]),
            # "deap_crossover_kwargs": hp.choice([{}]),

            "deap_mutation_operator": hp.choice("deap_mutation_operator1", [mutGaussian]),
            "deap_mutation_kwargs": hp.choice("deap_mutation_kwargs1", [{"mu": 0, "sigma": 1, "indpb": 0.3},
                                                                        {"mu": 0, "sigma": 1, "indpb": 0.6},
                                                                        {"mu": 0, "sigma": 1, "indpb": 0.2},
                                                                        {"mu": 0, "sigma": 1, "indpb": 0.8}]),

            "tournament_method": hp.choice("tournament_method1", [selTournament]),
            "tournament_kwargs": hp.choice("tournament_kwargs1", [{"k": 5, "tournsize": 10},
                                                                  {"k": 4, "tournsize": 8},
                                                                  {"k": 3, "tournsize": 8}]),
            "enemy_number": hp.choice("enemy_number1", [[1, 4, 7], [2, 4, 8], [1, 2], [5, 8]]),
            "survivor_pool": hp.choice("survivor_pool1", ['all', 'offspring']),
            "survivor_selection_method": hp.choice("survivor_selection_method1", [selProportional, selBest]),
        },
    ])}

# for i in range(5):
#     print(ho_sample(search_space))


# # We can run Hyperopt locally (only on the driver machine)
# # by calling `fmin` without an explicit `trials` argument.
# best_hyperparameters = fmin(
#     fn=train,
#     space=search_space,
#     algo=algo,
#     max_evals=5,
#     timeout=3600)
# print('best_hyperparameters', best_hyperparameters)




hyperopt_search = HyperOptSearch(search_space, metric="mean_accuracy", mode="max")

analysis = tune.run(train,
                    num_samples=8*1,
                    search_alg=hyperopt_search,
                    time_budget_s=3600*TUNING_HOURS,
                    verbose=3,
                    scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
                    # resources_per_trial={'gpu': 1},  # to enable GPU
                    )
df = analysis.results_df
prev_results = [path for path in os.listdir() if 'hyperopt_results' in path]
df.to_csv(f'hyperopt_results_{len(prev_results) + 1}.csv')

# Obtain a trial dataframe from all run trials of this `tune.run` call.
dfs = analysis.trial_dataframes
# Plot by epoch
ax = None  # This plots everything on the same plot
for d in dfs.values():
    ax = d.mean_accuracy.plot(ax=ax, legend=False)
ax.set_xlabel('Epochs')
ax.set_ylabel("Mean Accuracy")
ax.savefig(f'hyperopt_epochs_{len(prev_results) + 1}.png')

