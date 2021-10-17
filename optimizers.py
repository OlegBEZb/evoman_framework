import os
import sys
import numpy as np
from evoman.environment import Environment
from selection import selRandom, selProportional
from deap.tools.mutation import mutGaussian
from deap.tools.crossover import cxOnePoint
from custom_crossover import cx4ParentsCustomUniform, cxMultiParentUniform
import random
import csv

from utils import norm


class EvolutionaryAlgorithm:
    def __init__(self,
                 env: Environment,
                 experiment_name: str,
                 weight_amplitude=1,
                 mating_num=2,
                 population_size=100,
                 patience=15,
                 tournament_method=selRandom,
                 tournament_kwargs={"k": 2},
                 doomsday_population_ratio=0.25,
                 doomsday_replace_with_random_prob=0.9,
                 deap_mutation_operator=mutGaussian,
                 deap_mutation_kwargs={},
                 deap_crossover_method=cxOnePoint,
                 deap_crossover_kwargs={},
                 survivor_pool="all",  # can be all or offspring
                 survivor_selection_method=selProportional,
                 ):

        self.env = env

        self.experiment_name = experiment_name

        self.layer_sizes = self.env.player_controller.hidden_layer_sizes
        self.weights_num = (env.get_num_sensors() + 1) * self.layer_sizes[0] + (self.layer_sizes[0] + 1) * \
                           self.layer_sizes[1]

        self.weight_amplitude = weight_amplitude
        self.population_size = population_size
        self.mating_num = mating_num
        self.patience = patience

        self.tournament_method = tournament_method
        self.tournament_kwargs = tournament_kwargs

        self.doomsday_population_ratio = doomsday_population_ratio
        self.doomsday_replace_with_random_prob = doomsday_replace_with_random_prob

        self.deap_mutation_operator = deap_mutation_operator
        self.deap_mutation_kwargs = deap_mutation_kwargs
        self.deap_crossover_method = deap_crossover_method
        self.deap_crossover_kwargs = deap_crossover_kwargs

        self.survivor_pool = survivor_pool
        self.survivor_selection_method = survivor_selection_method

    def initialize_population(self):
        # initializes population loading old solutions or generating new ones

        if not os.path.exists(self.experiment_name + '/evoman_solstate'):
            print('NEW EVOLUTION')
            population = np.random.uniform(-self.weight_amplitude, self.weight_amplitude,
                                           (self.population_size, self.weights_num))
            population_fitness = self.evaluate_in_simulation(population)
            solutions = [population, population_fitness]
            self.env.update_solutions(solutions)

            self.write_to_csv('results.csv',
                              ['generation', 'best_score', 'fitness_mean', 'fitness_std'],
                              'w')

            best_individual_id, best_score, fitness_mean, fitness_std, msg = self.get_population_stats(
                population_fitness, generation_num=0)
            # saves results for first pop
            np.savetxt(os.path.join(self.experiment_name, 'best_solution.txt'),
                       population[best_individual_id])
            print(msg)
            self.write_to_csv('results.csv',
                              [0, best_score, fitness_mean, fitness_std],
                              'a')
            start_generation = 1  # as we have just sampled the 0th gen
        else:
            print('CONTINUING EVOLUTION')
            self.env.load_state()
            population = self.env.solutions[0]
            population_fitness = self.env.solutions[1]

            # finds last generation number
            file = open(os.path.join(self.experiment_name, 'results.csv'))
            reader = csv.reader(file)
            start_generation = len(list(reader)) - 1

            _, best_score, _, _, msg = self.get_population_stats(population_fitness,
                                                                 generation_num=start_generation - 1)

        return population, population_fitness, best_score, start_generation

    def write_to_csv(self, filename, row, mode):
        # open the file in the write mode
        with open(os.path.join(self.experiment_name, filename), mode, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    @staticmethod
    def simulation(env, x):
        fitness, player_life, enemy_life, time = env.play(pcont=x)
        return fitness

    @staticmethod
    def simulation_gain(env, x):
        fitness, player_life, enemy_life, time = env.play(pcont=x)
        return player_life - enemy_life

    def evaluate_in_simulation(self, x, gain_not_fitness=False):
        # simulates each of x
        if gain_not_fitness:
            return np.array(list(map(lambda y: self.simulation_gain(self.env, y), x)))
        else:
            return np.array(list(map(lambda y: self.simulation(self.env, y), x)))

    def crossover(self, population, population_fitness):
        """
        Crossover/recombination is a genetic operator used to vary the programming of a chromosome or chromosomes from one generation
        to the next. Crossover is sexual reproduction. Two strings are picked from the mating pool at random to
        crossover in order to produce superior offspring. The method chosen depends on the Encoding Method.

        Parameters
        ----------
        population_fitness
        population

        Returns
        -------

        """
        total_offspring = np.zeros((0, self.weights_num))

        for parents_set in range(0, population.shape[0],
                                 self.tournament_kwargs['k']):  # define a variable for the number of parents sets
            if self.tournament_method == selRandom:
                parents = self.tournament_method(population, **self.tournament_kwargs)
            else:
                parents = self.tournament_method(population, population_fitness, **self.tournament_kwargs)

            children = []
            for _ in range(self.mating_num):
                if self.deap_crossover_method in [cx4ParentsCustomUniform, cxMultiParentUniform]:
                    children.extend(self.deap_crossover_method(parents, **self.deap_crossover_kwargs))
                else:
                    children.extend(self.deap_crossover_method(parents[0], parents[1], **self.deap_crossover_kwargs))

            offspring = random.sample(children, random.randint(1, len(children)))
            offspring = self.mutation(offspring)
            # also converts list to ndarray (num_children, weigths_num)
            offspring = np.clip(offspring, -self.weight_amplitude, self.weight_amplitude)

            total_offspring = np.vstack((total_offspring, offspring))

        return total_offspring

    def mutation(self, offspring):
        # for i in range(len(offspring)):
        #     for j in range(len(offspring[i])):
        #         if np.random.uniform(0, 1) <= self.mutation_prob:
        #             offspring[i][j] = offspring[i][j] + np.random.normal(0, 1)
        result = []
        for individual in offspring:
            # returns a tuple
            result.append(self.deap_mutation_operator(individual, **self.deap_mutation_kwargs)[0])
        return result

    # kills the worst genomes, and replace with new best/random solutions
    def doomsday(self, population, population_fitness):
        worst_num = int(self.population_size * self.doomsday_population_ratio)  # a quarter of the population
        order = np.argsort(population_fitness)
        orderasc = order[0:worst_num]

        new_random_counter = 0
        for o in orderasc:
            for j in range(0, self.weights_num):
                # pro = np.random.uniform(0, 1)
                if np.random.uniform(0, 1) <= self.doomsday_replace_with_random_prob:
                    new_random_counter += 1
                    population[o][j] = np.random.uniform(-self.weight_amplitude,
                                                         self.weight_amplitude)  # random dna, uniform dist.
                else:
                    population[o][j] = population[order[-1:]][0][j]  # dna from best

            population_fitness[o] = self.evaluate_in_simulation([population[o]])

        print(
            f'During the doomsday {new_random_counter} genes were replaced with random ones and {worst_num * self.weights_num - new_random_counter} with the best one')

        return population, population_fitness

    def train(self, generations=30):

        population, population_fitness, best_score, start_generation = self.initialize_population()
        last_best_score = best_score
        gens_without_improvement = 0

        for i in range(start_generation, generations):

            offspring = self.crossover(population, population_fitness)
            offspring_fitness = self.evaluate_in_simulation(offspring)

            if self.survivor_pool == "all":
                population = np.vstack((population, offspring))
                population_fitness = np.append(population_fitness, offspring_fitness)
            elif self.survivor_pool == "offspring":
                population = offspring
                population_fitness = offspring_fitness
            else:
                raise ValueError('survivor_pool should be all or offspring ', self.survivor_pool, ' given')

            best_individual_id, best_score, _, _, _ = self.get_population_stats(population_fitness, i)

            if best_score <= last_best_score:
                gens_without_improvement += 1
            else:
                last_best_score = best_score
                gens_without_improvement = 0
                np.savetxt(os.path.join(self.experiment_name, 'best_solution.txt'), population[best_individual_id])

            # selection
            best_individual = population[best_individual_id]
            population = self.survivor_selection_method(population, population_fitness,
                                                        self.population_size - 1)
            population = np.stack(population, axis=0)
            population = np.vstack((population, best_individual))
            population_fitness = self.evaluate_in_simulation(population)

            if gens_without_improvement >= self.patience:
                # file_aux = open(self.experiment_name + '/results.txt', 'a')
                # file_aux.write('\ndoomsday')
                # file_aux.close()

                population, population_fitness = self.doomsday(population, population_fitness)
                gens_without_improvement = 0

            best_individual_id, best_score, fitness_mean, fitness_std, msg = self.get_population_stats(
                population_fitness, i)

            # saves results
            print(msg)
            self.write_to_csv('results.csv', [i, best_score, fitness_mean, fitness_std], 'a')

            # saves file with the best solution
            # w1 | w2 | ... | w255 | fitness | generation
            # np.savetxt(self.experiment_name + '/best_solution.txt', population[best_individual_id])

            # saves simulation state
            solutions = [population, population_fitness]
            self.env.update_solutions(solutions)
            self.env.save_state()

    @staticmethod
    def get_population_stats(population_fitness, generation_num=None):
        best_individual_id = np.argmax(population_fitness)
        best_score = np.max(population_fitness)
        fitness_mean = np.mean(population_fitness)
        fitness_std = np.std(population_fitness)

        if generation_num is not None:
            msg = f"GENERATION {generation_num}: "
        else:
            msg = ''
        msg += f"best score: {round(best_score, 3)} mean fitness: {round(fitness_mean, 3)} fitness std: {round(fitness_std, 3)}"

        return best_individual_id, best_score, fitness_mean, fitness_std, msg

    def test(self, n_times=5, gain_not_fitness=False):
        # try:
        #     del os.environ['SDL_VIDEODRIVER']
        # except:
        #     raise
        # loads file with the best solution for testing
        fitness_list = []
        best_solution = np.loadtxt(self.experiment_name + '/best_solution.txt')
        for i in range(n_times):
            print('RUNNING SAVED BEST SOLUTION')
            self.env.update_parameter('speed', 'normal')
            self.env.update_parameter('visualmode', 'yes')
            fitness = self.evaluate_in_simulation([best_solution], gain_not_fitness=gain_not_fitness)
            fitness_list.append(fitness)

        np.savetxt(os.path.join(self.experiment_name, 'test_results.txt'), fitness_list)
        return fitness_list
