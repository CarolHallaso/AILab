import random
import sys
import math

import numpy as np

POPSIZE = 2048  # genome population size
MAXITER = 16384  # maximum iterations (generations)
ELITRATE = 0.1  # elitism rate
MUTATIONRATE = 0.25  # mutation rate
MUTATION = sys.maxsize * MUTATIONRATE

class GA_struct:

    # Citizens of our Population
    def __init__(self, string, fitness):
        self.string = string
        self.fitness = fitness

class BinPacking:

    def __init__(self, objects, Bsize):
        self.NumOfObjects = len(objects)
        self.Bsize = Bsize
        self.objects = objects

    def init_population(self, population: list, buffer: list):

        for i in range(POPSIZE):
            citizen = GA_struct("", 0)
            citizen.string = np.random.permutation(self.NumOfObjects)
            population[i] = citizen
        return

    def calc_fitness(self, population):
        c = self.Bsize
        n = self.NumOfObjects
        k = 2
        N = len(set(self.objects))

        for i in range(self.NumOfObjects):
            fitness = pow((self.objects[i] / c), k)
            fitness = fitness / N
            population[i].fitness = fitness
        return

    def sort_by_fitness(self, population):
        population.sort(key=lambda x: x.fitness)
        return

    def print_best(self, population):
        print("Best: " + population[0].string + " fitness: " + " (" + str(population[0].fitness) + ")")
        return

    def elitism(self, population: list[GA_struct], buffer: list[GA_struct], esize):
        temp = population[:esize].copy()
        buffer[:esize] = temp
        return

    def mutate(self, member: GA_struct):

        t_size = len(member.string)
        ipos = random.randrange(0, t_size - 1)
        delta = random.randrange(0, 90) + 32
        string = member.string[: ipos] + chr((ord(member.string[ipos]) + delta) % 122) + member.string[ipos + 1:]
        member.string = string
        return

    def mate(self, population: list[GA_struct], buffer: list[GA_struct], type):

        esize = int(POPSIZE * ELITRATE)
        self.elitism(population, buffer, esize)

        # mate the rest
        for i in range(esize, POPSIZE):

            i1 = random.randrange(0, POPSIZE // 2)
            i2 = random.randrange(0, POPSIZE // 2)

            if type == "SINGLE":
                pos = random.randrange(0, esize)
                buffer[i] = GA_struct(population[i1].string[0: pos] + population[i2].string[pos:], 0)

            elif type == "DOUBLE":
                pos1 = random.randrange(0, esize - 2)
                pos2 = random.randrange(pos1 + 1, esize - 1)
                buffer[i] = GA_struct(
                    population[i1].string[0: pos1] + population[i2].string[pos1:pos2] + population[i1].string[pos2:], 0)

            elif type == "UNIFORM":
                gen = ""
                for j in range(esize):
                    r = random.randrange(0, 2)
                    if r == 0:
                        gen = gen + population[i1].string[j]

                    else:
                        gen = gen + population[i2].string[j]
                buffer[i] = GA_struct(gen, 0)

            if random.randrange(sys.maxsize) < MUTATION:
                self.mutate(buffer[i])

        return

    def swap(self, population: list[GA_struct], buffer: list[GA_struct]):

        return buffer, population


# Returns number of bins required using first fit algorithm
def firstFit(weights, Bins_capacity):
    # Initialize result (Count of bins)
    result = 0
    num_of_objects = len(weights)
    # Create an array to store remaining space in bins
    # there can be at most n bins
    bin_rem = [0] * num_of_objects

    for i in range(num_of_objects):
        # Find the first bin that has enough space
        j = 0
        while j < result:
            if bin_rem[j] >= weights[i]:
                bin_rem[j] = bin_rem[j] - weights[i]
                break
            j += 1

        # If no bin could accommodate weight[i]
        if j == result:
            bin_rem[result] = Bins_capacity - weights[i]
            result = result + 1
    return result


if __name__ == "__main__":

    weights = [2, 5, 4, 7, 1, 3, 8]
    c = 10

    print("Number of bins required in First Fit : ", firstFit(weights,c))

    problem = BinPacking(weights, c)
    random.seed()
    pop_alpha = [None] * POPSIZE
    pop_beta = [None] * POPSIZE
    problem.init_population(pop_alpha, pop_beta)
    population = pop_alpha
    buffer = pop_beta

    for i in range(MAXITER):
        problem.calc_fitness(population)
        problem.sort_by_fitness(population)
        print("num of required bins according to genetic algorithm")
        problem.print_best(population)

        problem.mate(population, buffer, "UNIFORM")
        population, buffer = problem.swap(population, buffer)
