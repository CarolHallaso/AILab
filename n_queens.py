import copy
import math
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
e = 2.71828182846

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
        # to mutate we choose a position randomly and change it to a random character
        target_size = QUEENS
        ipos = randint(0, target_size - 1)
        delta = randint(0, QUEENS - 1)
        agent.string = agent.string[:ipos] + [(agent.string[ipos] + delta) % QUEENS] + agent.string[ipos + 1:]

    def mate(self, population, buffer, cross_over_type, selection_method=None, probabilities=None):
        esize = int(GA_POPSIZE * GA_ELITRATE)
        target_size = QUEENS
        self.elitism(population, buffer, esize)
        # we choose the parents depending on the method specified by the given input "selection_method"
        # and to each pair of parents we do a cross over according to the given input "cross_over_type"
        parents_idx = 0
        if selection_method == "sus":
            parents = self.sus(population, GA_POPSIZE - esize)

        # mate the rest
        for i in range(esize, GA_POPSIZE):

            if selection_method == None:
                i1 = random.randrange(0, GA_POPSIZE // 2)
                i2 = random.randrange(0, GA_POPSIZE // 2)

            if i != (GA_POPSIZE - 1) and selection_method == "sus":
                i1 = parents[parents_idx]
                i2 = parents[parents_idx + 1]
                parents_idx += 1

            if selection_method == "rws":  # we choose the parents according to the RWS algo we learned
                i1 = self.rws(population, probabilities)
                i2 = self.rws(population, probabilities)

            if selection_method == "tournament":  # we choose the parents according to the tournament selection algo we learned
                k = 3
                i1, i2 = self.tournament_selection(population, k)

            # we choose the parents according to the minimal conflict heuristic
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
        # calculates and returns the average fitness of the population
        sum = 0

        for i in range(len(population)):
            sum += population[i].fitness

        return sum / GA_POPSIZE

    def calcStd(self):
        # calculates and returns the STD of the population
        fitness = []

        for i in range(len(population)):
            fitness.append(population[i].fitness)

        return statistics.stdev(fitness)

    def rws(self, population, probabilities):
        # Roulette wheel selection algorithm
        # implements the pseudocode we saw in class
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
        # Stochastic universal sampling
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
        # we randomly choose a sample from the population with size k and choose the two best ones
        sample = []
        for i in range(k):
            sample.append(population[random.randrange(0, len(population) - 1)])

        sample.sort(key=lambda x: x.fitness)
        return sample[0], sample[1]

    @staticmethod
    def positive_random(rng, conflicts, filter):
        return random.choice([i for i in range(rng) if filter(conflicts[i])])

    def init_roulette(self, population):
        # build the roulette with the probabilities being according to the fitnesses
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
        # the minimal conflict heuristic chooses solution that minimizes the conflicts(collisions)
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

    def kendall_tau_distance(self, values1, values2):
        """Compute the Kendall tau distance."""
        n = len(values1)
        assert len(values2) == n, "Both lists have to be of equal length"
        i, j = np.meshgrid(np.arange(n), np.arange(n))
        a = np.argsort(values1)
        b = np.argsort(values2)
        ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]),
                                    np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
        return ndisordered / 2

    def calculate_distance(self, first, second):

        diff = problem.kendall_tau_distance(first.permutation, second.permutation)

        return diff

    def get_num_of_genomes_with_fitness(self, population, fitness):
        i = 0
        num = 0
        for i in range(len(population)):
            if population[i].fitness == fitness:
                num += 1
                i += 1

        return num

    def calculate_genetic_diversity(self, population):
        ans = 0
        for i in range(len(population)):

            for j in range(len(population)):
                ans += self.calculate_distance(population[i], population[j])
        return ans / len(population)


if __name__ == "__main__":

    problem = GeneticAlgorithm()
    random.seed()
    pop_alpha = [None] * GA_POPSIZE
    pop_beta = [None] * GA_POPSIZE

    population = problem.initPop(pop_alpha, pop_beta)  #initialize the population

    buffer = pop_beta
    start_t = time.time()  # starting time


    probabilities = problem.init_roulette(population)

    generation_num = 0

    for i in range(GA_MAXITER):

        time2 = time.time()  # clock ticks

        generation_num += 1

        problem.calc_fitness_queens(population)
        population.sort(key=lambda x: x.fitness)  # sort population array by fitness
        print("Best:", population[0].string, "fitness: ", population[0].fitness)  # print string with best fitness
        print("mean of generation is: " + str(problem.calcAVG()))
        print("standard deviation of generation is: " + str(problem.calcStd()))

        clock_ticks = time.time() - time2
        E_T = time.time() - start_t
        print("Clock ticks: " + str(clock_ticks))
        print("Elapsed time: " + str(E_T))

        best_fitness = population[0].fitness
        number_of_best_genomes = problem.get_num_of_genomes_with_fitness(population, best_fitness)
        mid = len(population) / 2
        mid = math.floor(mid)
        mid_fitness = population[mid].fitness
        num_of_mid_genomes = problem.get_num_of_genomes_with_fitness(population, mid_fitness)
        print("best fitness = " + str(best_fitness))
        print("num of best fitness = " + str(number_of_best_genomes))
        print("mid fitness = " + str(mid_fitness))
        print("num of mid fitness = " + str(num_of_mid_genomes))
        prob_best = number_of_best_genomes / len(population)
        prob_mid = num_of_mid_genomes / len(population)
        selection_pressure = 0
        if prob_mid != 0:
            selection_pressure = prob_best / prob_mid

        print("selection pressure:" + str(selection_pressure))

        # genetic_diversity = self.calculate_genetic_diversity(population)

        # print("Genetic Diversity: " + str(genetic_diversity))

        # fitness = []
        # for j in range(len(population)):
        #     fitness.append(population[j].fitness)

        # plt.xlabel('Fitness')
        # plt.ylabel('Number of Genomes')
        # plt.hist(fitness)
        # plt.show()

        if population[0].fitness == 0:
            break

        buffer = problem.mate(population, buffer, "DOUBLE", "tournament")  # mate
        population, buffer = buffer, population

        for genome in population:
            genome.age += 1

            # uniform decay mutation
            # rate = GA_MUTATIONRATE * (1 / GA_MAXITER)
            # GA_MUTATIONRATE = GA_MUTATIONRATE - rate

            # Adaptive decrease function mutation
            pmax = 0.3
            r = 0.5
            helper1 = 2 * (pmax ** 2) * (e ** (r * generation_num))
            helper2 = pmax + (pmax * (e ** (r * generation_num)))
            GA_MUTATIONRATE = helper1 / helper2

        E_T = time.time() - start_t
        clock_ticks = time.time() - time2
        print("Elapsed time: " + str(E_T) + " Clock Ticks: " + str(clock_ticks))