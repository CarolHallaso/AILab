import copy
import operator
import sys
import random
import statistics
import time
import timeit

from matplotlib import pyplot as plt
import numpy as np
from numpy import cumsum, resize
from numpy.random import randint

GA_POPSIZE = 2048  # genome population size
GA_MAXITER = 16384  # maximum iterations (generations)
GA_ELITRATE = 0.1  # elitism rate
GA_MUTATIONRATE = 0.25  # mutation rate
GA_MUTATION = sys.maxsize * GA_MUTATIONRATE
QUEENS = 8

class GA_struct:

    # Citizens of our Population
    def __init__(self, string, fitness):
        self.string = string
        self.fitness = fitness
        self.age = 0


class GeneticAlgorithm:
    def initPop(self, population, buffer):
        # initialize population
        for i in range(GA_POPSIZE):
            randArray = [randint(0, QUEENS - 1) for i in range(QUEENS)]  # random numbers array
            x = GA_struct(randArray, 0)
            population[i] = x
            #self.buffer.append(GA_struct([], 0))

        return population

    def calc_fitness_queens(self, population):
        # num of queens that collide
        for agent in population:
            count = [0] * QUEENS
            bad_count = 0
            for j in agent.string:  # count row collisions
                count[int(j)] += 1
            for x in count:
                if x != 1:  # if collide
                    bad_count += 1
            for i in range(QUEENS):
                for j in range(QUEENS):
                    if i == j:
                        continue
                    if abs(i - j) == abs(int(agent.string[i]) - int(agent.string[j])):
                        bad_count += 1

            agent.fitness = bad_count

    def elitism(self, population, buffer, esize):
        temp = population[:esize].copy()
        buffer[:esize] = temp
        return

    def mutate(self, agent):
        target_size = QUEENS
        ipos = randint(0, target_size - 1)
        delta = randint(0, QUEENS - 1)
        agent.string = agent.string[:ipos] + [(agent.string[ipos] + delta) % QUEENS] + agent.string[ipos + 1:]

    def mate(self, population, buffer, cross_over_type, selection_method=None, probabilities=None):
        esize = int(GA_POPSIZE * GA_ELITRATE)
        target_size = QUEENS
        self.elitism(population, buffer, esize)

        parents_idx = 0
        if selection_method == "sus":
            parents = self.sus(population, GA_POPSIZE - esize)

        for i in range(esize, GA_POPSIZE):

            if selection_method == None:
                i1 = random.randrange(0, GA_POPSIZE // 2)
                i2 = random.randrange(0, GA_POPSIZE // 2)

            if i != (GA_POPSIZE - 1) and selection_method == "sus":
                i1 = parents[parents_idx]
                i2 = parents[parents_idx + 1]
                parents_idx += 1

            if selection_method == "rws":
                i1 = self.rws(population, probabilities)
                i2 = self.rws(population, probabilities)

            if selection_method == "tournament":
                k = 3
                i1, i2 = self.tournament_selection(population, k)

            if selection_method == "min_conf" and i<GA_POPSIZE-1:
                i1 = self.miniConf(population[i].string, QUEENS, iters=100)
                i2 = self.miniConf(population[i+1].string, QUEENS, iters=100)
            if cross_over_type == "SINGLE":
                pos = random.randrange(1, target_size)
                if selection_method == "min_conf":
                    buffer[i] = GA_struct(i1[0: pos] + i2[pos:], 0)

                elif selection_method != None and selection_method != "min_conf":
                    buffer[i] = GA_struct(i1.string[0: pos] + i2.string[pos:], 0)
                else:
                    buffer[i] = GA_struct(population[i1].string[0: pos] + population[i2].string[pos:], 0)

            elif cross_over_type == "DOUBLE":
                pos1 = random.randrange(0, target_size - 2)
                pos2 = random.randrange(pos1 + 1, target_size - 1)
                if selection_method == "min_conf":
                    buffer[i] = GA_struct(i1[0: pos1] + i2[pos1:pos2] + i1[pos2:], 0)

                elif selection_method != None and selection_method != "min_conf":
                    buffer[i] = GA_struct(i1.string[0: pos1] + i2.string[pos1:pos2] + i1.string[pos2:], 0)
                else:
                    buffer[i] = GA_struct(
                        population[i1].string[0: pos1] + population[i2].string[pos1:pos2] + population[i1].string[pos2:], 0)

            elif cross_over_type == "UNIFORM":
                gen = []
                for j in range(target_size):
                    r = random.randrange(0, 2)
                    if r == 0:
                        if selection_method == "min_conf":
                            gen.append(i1[j])

                        elif selection_method != None and selection_method != "min_conf":
                            gen.append(i1.string[j])
                        else:
                            gen.append(population[i1].string[j])

                    else:
                        if selection_method == "min_conf":
                            gen.append(i2[j])

                        elif selection_method != None and selection_method != "min_conf":
                            gen.append(i2.string[j])
                        else:
                            gen.append(population[i2].string[j])
                buffer[i] = GA_struct(gen, 0)

            if random.randrange(sys.maxsize) < GA_MUTATION:
                self.mutate(buffer[i])

        return buffer

    def calcAVG(self):

        sum = 0

        for i in range(len(population)):
            sum += population[i].fitness

        return sum / GA_POPSIZE

    def calcStd(self):

        fitness = []

        for i in range(len(population)):
            fitness.append(population[i].fitness)

        return statistics.stdev(fitness)

    def rws(self, population, probabilities):
        # Roulette wheel selection algorithm
        rndNumber = random.random()
        offset = 0.0
        for i in range(GA_POPSIZE):
            offset += probabilities[i]
            if rndNumber < offset:
                return population[i]

    def get_subset_sum(self, population, index):
        sum = 0
        for i in range(index):
            sum += population[i].fitness
        return sum

    def sus(self, population, N):
        sum = 0
        for i in range(len(population)):
            sum += 1 / population[i].fitness if population[i].fitness else 0
        point_distance = sum / N
        start_point = random.uniform(0, point_distance)
        points = [start_point + i * point_distance for i in range(N)]
        parents = set()
        while len(parents) < N:
            random.shuffle(population)
            i = 0
            while i < len(points) and len(parents) < N:
                j = 0
                while j < len(population):
                    if self.get_subset_sum(population, j) < points[i]:
                        parents.add(population[j])
                        break
                    j += 1
                i += 1

        return list(parents)

    def tournament_selection(self, population, k):
        sample = []
        for i in range(k):
            sample.append(population[random.randrange(0, len(population) - 1)])

        sample.sort(key=lambda x: x.fitness)
        return sample[0], sample[1]

    @staticmethod
    def positive_random(rng, conflicts, filter):
        return random.choice([i for i in range(rng) if filter(conflicts[i])])

    def inversion_mutation(self):
        index1 = random.randrange(0, len(self) - 1)
        index2 = random.randrange(index1, len(self))
        size = index2 - index1
        while i <= size / 2:
            tmp = self[index1]
            self[index1] = self[index2]
            self[index2] = tmp
        return self

    def scramble_mutation(self):
        index1 = random.randrange(0, len(self) - 1)
        index2 = random.randrange(index1, len(self))
        size = index2 - index1
        helper = self[index1:index2 + 1]
        random.shuffle(helper)
        i = index1
        j = 0
        while i < index2 + 1:
            self[i] = helper[j]
            j += 1
        return self

    def init_roulette(self, population):
        problem.calc_fitness_queens(population)
        probs = []
        total_fitness = 0
        for i in range(GA_POPSIZE):
            total_fitness += population[i].fitness
        for i in range(GA_POPSIZE):
            probs.append(population[i].fitness / total_fitness)
        return probs

    def random_position(self, li, filter, num_rows):
        return random.choice([i for i in range(num_rows) if filter(li[i])])

    def miniConf(self, solution, num_rows, iters=100):
        for k in range(iters):
            confs = self.find_conflicts(solution, num_rows)
            if sum(confs) == 0:
                return solution
            col = self.random_position(confs, lambda elt: elt > 0, num_rows)
            vconfs = [self.hits(solution, num_rows, col, row) for row in range(num_rows)]
            solution[col] = self.random_position(vconfs, lambda elt: elt == min(vconfs), num_rows)

        return solution

    def find_conflicts(self, solution, num_rows):
        res = []
        for row in range(num_rows):
            res.append(self.hits(solution, num_rows, row, solution[row]))
        return res

    def hits(self, solution, num_rows, col, row):
        total = 0
        for i in range(num_rows):
            if i == col:
                continue
            if solution[i] == row or abs(i - col) == abs(solution[i] - row):
                total += 1
        return total


if __name__ == "__main__":

    problem = GeneticAlgorithm()
    random.seed()
    pop_alpha = [None] * GA_POPSIZE
    pop_beta = [None] * GA_POPSIZE

    population = problem.initPop(pop_alpha, pop_beta)

    buffer = pop_beta
    start_t = time.time()


    probabilities = problem.init_roulette(population)

    for i in range(GA_MAXITER):

        time2 = time.time()  # clock ticks

        problem.calc_fitness_queens(population)
        population.sort(key=lambda x: x.fitness)  # sort population array by fitness
        print("Best:", population[0].string, "fitness: ", population[0].fitness)  # print string with best fitness
        print("mean of generation is: " + str(problem.calcAVG()))
        print("standard deviation of generation is: " + str(problem.calcStd()))

        clock_ticks = time.time() - time2
        E_T = time.time() - start_t
        print("Clock ticks: " + str(clock_ticks))
        print("Elapsed time: " + str(E_T))

        # fitness = []
        # for j in range(len(population)):
        #     fitness.append(population[j].fitness)

        # plt.xlabel('Fitness')
        # plt.ylabel('Number of Genomes')
        # plt.hist(fitness)
        # plt.show()

        if population[0].fitness == 0:
            break

        buffer = problem.mate(population, buffer, "DOUBLE", "rws", probabilities)  # mate
        population, buffer = buffer, population

        for genome in population:
            genome.age += 1

        E_T = time.time() - start_t
        clock_ticks = time.time() - time2
        print("Elapsed time: " + str(E_T) + " Clock Ticks: " + str(clock_ticks))
