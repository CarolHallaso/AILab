
import random
import sys

import numpy as np
from matplotlib import pyplot as plt

ELITRATE = 0.1  # elitism rate
MUTATIONRATE = 0.25  # mutation rate
MUTATION = sys.maxsize * MUTATIONRATE

class GA_struct:

    # Citizens of our Population
    def __init__(self, permutation, fitness):
        self.permutation = permutation
        self.fitness = fitness

class BaldwinExperiment:

    def __init__(self, size, iterations, bits):
        self.P_size = size
        self.Iterations = iterations
        self.Bits = bits
        self.population = [GA_struct([], 0)] * self.P_size
        self.pattern = ""

    def initialize_pattern(self):
        str = ""
        for i in range(self.Bits):
            helper = random.randrange(0, 2)
            if helper == 0:
                str += '0'
            else:
                str += '1'

        self.pattern = str

    def initialize_bit(self):
        helper = random.randrange(0, 4)
        if helper == 0:
            x = '0'
        elif helper == 1:
            x = '1'
        else:
            x = '?'
        return x

    def init_genome(self):
        str = ""
        for i in range(self.Bits):
            str += self.initialize_bit()

        return str

    def init_population(self, population):
        for i in range(len(population)):
            citizen = GA_struct([], 0)
            citizen.permutation = self.init_genome()
            population[i] = citizen
        return population

    def random_replace(self):
        r = random.randrange(0, 2)
        if r == 0:
            x = '0'
        else:
            x = '1'
        return x

    def count_correct_positions(self, target, permutation):
        count = 0
        for i in range(len(permutation)):
            x = target[i]
            y = permutation[i]
            if y == x:
                count += 1

        return count

    def count_incorrect_positions(self, target, permutation):
        count = 0
        for i in range(len(permutation)):
            x = target[i]
            y = permutation[i]
            if y != "?":
                if x != y:
                    count += 1

        return count

    def count_learned(self, permutation):
        count = 0
        for i in range(len(permutation)):
            y = permutation[i]
            if y == '?':
                count += 1

        return count

    def sort_by_fitness(self, population: list):
        population.sort(key=lambda x: x.fitness)
        return population

    def elitism(self, population: list[GA_struct], buffer: list[GA_struct], size):
        tmp = population[:size].copy()
        buffer[:size] = tmp
        return

    def RWS(self, population):
        sum_of_all_fitness = 0
        #for i in range(len(population)):
        #    sum_of_all_fitness += population[i].fitness
        #genomes_probabilities = [0] * len(population)
        #for j in range(len(population)):
        #    if sum_of_all_fitness > 0:
        #        genomes_probabilities[j] = population[j].fitness / sum_of_all_fitness
        #chosen = self.rws(population, genomes_probabilities)

        for i in range(len(population)):
            sum_of_all_fitness += population[i].fitness
        #sum_of_all_fitness = sum([genome.fitness for genome in population])
        if sum_of_all_fitness == 0:
            print(population[0].fitness)
        genomes_probabilities = [genome.fitness / sum_of_all_fitness for genome in population]

        return np.random.choice(population, p=genomes_probabilities)

    def mate(self, population, buffer: list[GA_struct]):
        size = int(self.P_size * ELITRATE)
        self.elitism(population, buffer, size)
        tsize = len(self.pattern)
        indx1 = -1
        indx2 = -2
        for i in range(len(population)):
            indx1 = self.RWS(population)
            indx2 = self.RWS(population)
            #indx1 = random.randrange(0, self.P_size // 2)
            #indx2 = random.randrange(0, self.P_size // 2)

            # single crossover
            pos = random.randrange(0, tsize)
            buffer[i] = GA_struct(indx1.permutation[0: pos] + indx2.permutation[pos:], 0)
            #buffer[i] = population[indx1]

        return buffer, population

    def local_search(self, population, correct, incorrect, learned):

        fitness = [0] * self.P_size
        print(self.pattern)

        for i in range(self.P_size):  # iterate over the population
            tmp_iteration = self.Iterations
            inner_correct = [0] * 1000
            inner_incorrect = [0] * 1000
            for j in range(self.Iterations):  # for each genome iterate over max iterations
                inner_correct[j] = self.count_correct_positions(self.pattern, population[i].permutation) / len(self.pattern)
                inner_incorrect[j] = self.count_incorrect_positions(self.pattern, population[i].permutation) / len(self.pattern)

                str = ""
                old_str = population[i].permutation
                for k in range(self.Bits):
                    if population[i].permutation[k] == '?':
                        # if it is '?' then make it 0 with p = 0.5 and 1 with p = 0.5
                       str += self.random_replace()
                    else:
                        str += population[i].permutation[k]

                population[i].permutation = str

                if i == 1:
                    print(population[i].permutation)
                if population[i].permutation == self.pattern:
                    tmp_iteration = j + 1
                    print("found")
                    print(population[i].permutation)
                    break
                population[i].permutation = old_str

            correct[i] = sum(inner_correct) / tmp_iteration
            incorrect[i] = sum(inner_incorrect) / tmp_iteration
            learned[i] = self.count_learned(population[i].permutation) / len(self.pattern)

            n = self.Iterations - tmp_iteration
            fitness[i] = 1 + (19 * n / self.Iterations)  # fitness = 1 + 19n/1000
            population[i].fitness = fitness[i]
        return max(fitness), fitness, population, correct, incorrect, learned


if __name__ == "__main__":
    problem = BaldwinExperiment(1000, 1000, 20)
    num_of_generations = 40
    pop1 = [None] * 1000
    pop2 = [None] * 1000
    buffer = pop2
    problem.initialize_pattern()
    population = problem.init_population(pop1)
    correct = [0] * len(population)
    incorrect = [0] * len(population)
    learned = [0] * len(population)
    x = [0] * num_of_generations
    AVG_correct_gen = [0] * num_of_generations
    AVG_incorrect_gen = [0] * num_of_generations
    AVG_learned_gen = [0] * num_of_generations

    for i in range(num_of_generations):
        x[i] = i
        print(i)

        population = problem.sort_by_fitness(population)
        #fitness = [0] * problem.P_size

        max_fitness, fitness, population, correct, incorrect, learned = problem.local_search(population, correct, incorrect, learned)
        Avg_correct = sum(correct) / len(correct)
        Avg_incorrect = sum(incorrect) / len(incorrect)
        Avg_learned = sum(learned) / len(learned)
        print(fitness)
        print(max_fitness)

        population, buffer = problem.mate(population, buffer)
        print(Avg_correct)
        print(Avg_incorrect)
        print(Avg_learned)
        AVG_correct_gen[i] = Avg_correct
        AVG_incorrect_gen[i] = Avg_incorrect
        AVG_learned_gen[i] = Avg_learned

plt.xlabel('Generation')
plt.ylabel('Average percentage')
plt.title('Relationship between average percentage of correct/incorrect positions, learned bits and generation number')
plt.plot(x, AVG_correct_gen, label="correct posotions")
plt.plot(x, AVG_incorrect_gen, label="incorrect positions")
plt.plot(x, AVG_learned_gen, label="learned bits")
plt.legend()
plt.show()

    #print(problem.local_search())
