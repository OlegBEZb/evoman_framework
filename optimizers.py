import os, sys
import numpy as np
from evoman.environment import Environment
from deap.tools.crossover import cxOnePoint
from deap.tools.mutation import mutGaussian
import random


class EvolutionaryAlgorithm:
    def __init__(self,
                 env: Environment,
                 experiment_name: str,
                 weight_amplitude=1,
                 mating_num=2,
                 population_size=100,
                 mutation=0.2,
                 patience=15):

        self.env = env

        self.experiment_name = experiment_name

        # # number of weights for multilayer with 10 hidden neurons
        self.layer_size = self.env.player_controller.hidden_layer_sizes[0]
        self.weights_num = (env.get_num_sensors() + 1) * self.layer_size + (self.layer_size + 1) * 5

        self.weight_amplitude = weight_amplitude
        self.population_size = population_size
        self.mating_num = mating_num
        self.mutation_prob = mutation
        self.patience = patience

    def initialize_population(self):
        # initializes population loading old solutions or generating new ones

        if not os.path.exists(self.experiment_name + '/evoman_solstate'):
            print('NEW EVOLUTION')
            population = np.random.uniform(-self.weight_amplitude, self.weight_amplitude,
                                           (self.population_size, self.weights_num))
            population_fitness = self.evaluate_in_simulation(population)
            start_generation = 0
            solutions = [population, population_fitness]
            self.env.update_solutions(solutions)

            file_aux = open(self.experiment_name + '/results.txt', 'w') # new file
            file_aux.write('\n\ngen best mean std')
        else:
            print('CONTINUING EVOLUTION')
            self.env.load_state()
            population = self.env.solutions[0]
            population_fitness = self.env.solutions[1]

            # finds last generation number
            file_aux = open(self.experiment_name + '/gen.txt', 'r')
            start_generation = int(file_aux.readline()) + 1
            file_aux.close()

            file_aux = open(self.experiment_name + '/results.txt', 'a')


        best_individual_id, best_score, fitness_mean, fitness_std, msg = self.get_population_stats(population_fitness, 0)
        # saves results for first pop
        print(msg)
        file_aux.write(msg)
        file_aux.close()

        return population, population_fitness, start_generation

    @staticmethod
    def simulation(env, x):
        fitness, player_life, enemy_life, time = env.play(pcont=x)
        return fitness

    @staticmethod
    def norm(x, pfit_pop):
        if (max(pfit_pop) - min(pfit_pop)) > 0:
            x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
        else:
            x_norm = 0

        if x_norm <= 0:
            x_norm = 0.0000000001
        return x_norm

    def evaluate_in_simulation(self, x):
        # simulates each of x
        return np.array(list(map(lambda y: self.simulation(self.env, y), x)))

    def tournament(self, population, population_fitness):
        """
        Randomly picks two individuals from the population and returns the better

        Parameters
        ----------
        population_fitness
        population

        Returns
        -------

        """
        idx1 = np.random.randint(0, population.shape[0])
        idx2 = np.random.randint(0, population.shape[0])

        if population_fitness[idx1] > population_fitness[idx2]:
            return population[idx1]
        else:
            return population[idx2]

    def crossover(self, population, population_fitness):
        """
        Crossover is a genetic operator used to vary the programming of a chromosome or chromosomes from one generation
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

        for p in range(0, population.shape[0], 2):
            parent_1 = self.tournament(population, population_fitness)
            parent_2 = self.tournament(population, population_fitness)

            children = []
            for _ in range(self.mating_num):
                children.extend(cxOnePoint(parent_1, parent_2))

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
            result.append(mutGaussian(individual, 0, 1, self.mutation_prob)[0])
        return result

    # kills the worst genomes, and replace with new best/random solutions
    # TODO: reduce to the size of npop
    def doomsday(self, population, population_fitness):
        worst = int(self.population_size / 4)  # a quarter of the population
        order = np.argsort(population_fitness)
        orderasc = order[0:worst]

        for o in orderasc:
            for j in range(0, self.weights_num):
                pro = np.random.uniform(0, 1)
                if np.random.uniform(0, 1) <= pro:
                    population[o][j] = np.random.uniform(-self.weight_amplitude,
                                                         self.weight_amplitude)  # random dna, uniform dist.
                else:
                    population[o][j] = population[order[-1:]][0][j]  # dna from best

            population_fitness[o] = self.evaluate_in_simulation([population[o]])

        return population, population_fitness

    def train(self, generations=30):

        population, population_fitness, start_generation = self.initialize_population()
        print("Start generation:", start_generation)

        _, best_score, _, _, _ = self.get_population_stats(population_fitness)
        last_best_score = best_score
        gens_without_improvement = 0

        for i in range(start_generation, generations):

            offspring = self.crossover(population, population_fitness)
            offspring_fitness = self.evaluate_in_simulation(offspring)
            population = np.vstack((population, offspring))
            population_fitness = np.append(population_fitness, offspring_fitness)

            best_individual_id, best_score, _, _, _ = self.get_population_stats(population_fitness, i)

            # selection
            fit_pop_cp = population_fitness
            fit_pop_norm = np.array(list(map(lambda y: self.norm(y, fit_pop_cp),
                                             population_fitness)))  # avoiding negative probabilities, as fitness is ranges from negative numbers
            probs = (fit_pop_norm) / (fit_pop_norm).sum()
            selected_indices = np.random.choice(population.shape[0], self.population_size - 1, p=probs, replace=False)
            selected_indices = np.append(selected_indices, best_individual_id)
            population = population[selected_indices]
            population_fitness = population_fitness[selected_indices]

            # searching new areas

            if best_score <= last_best_score:
                gens_without_improvement += 1
            else:
                last_best_score = best_score
                gens_without_improvement = 0

            if gens_without_improvement >= self.patience:
                file_aux = open(self.experiment_name + '/results.txt', 'a')
                file_aux.write('\ndoomsday')
                file_aux.close()

                population, population_fitness = self.doomsday(population, population_fitness)
                gens_without_improvement = 0

            best_individual_id, best_score, fitness_mean, fitness_std, msg = self.get_population_stats(population_fitness, i)

            # saves results
            file_aux = open(self.experiment_name + '/results.txt', 'a')
            print(msg)
            file_aux.write('\n' + msg)
            file_aux.close()

            # saves generation number
            file_aux = open(self.experiment_name + '/gen.txt', 'w')
            file_aux.write(str(i))
            file_aux.close()

            # saves file with the best solution
            np.savetxt(self.experiment_name + '/best_solution.txt', population[best_individual_id])

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
    
    def test(self):
        # try:
        #     del os.environ['SDL_VIDEODRIVER']
        # except:
        #     raise
        # loads file with the best solution for testing
        best_solution = np.loadtxt(self.experiment_name + '/best_solution.txt')
        print('RUNNING SAVED BEST SOLUTION')
        self.env.update_parameter('speed', 'normal')
        self.env.update_parameter('visualmode', 'yes')
        self.evaluate_in_simulation([best_solution])

        sys.exit(0)
