import operator
import sys
import random
import statistics
import time
import timeit

from matplotlib import pyplot as plt
import numpy as np
from numpy import cumsum, resize

GA_POPSIZE = 2048  # genome population size
GA_MAXITER = 16384  # maximum iterations (generations)
GA_ELITRATE = 0.1  # elitism rate
GA_MUTATIONRATE = 0.25  # mutation rate
GA_MUTATION = sys.maxsize * GA_MUTATIONRATE
GA_TARGET = "Hello world! "

class GA_struct:

    # Citizens of our Population
    def __init__(self, string, fitness):
        self.string = string
        self.fitness = fitness
        self.age = 0


class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.personal_best = position
        self.velocity = velocity

    def velocity_update(self, w, c1, c2, global_best):
        for i in range(len(self.velocity)):
            cognitive = ord(self.personal_best[i]) - ord(self.position[i])
            social = ord(global_best[i]) - ord(self.position[i])
            self.velocity[i] = w * self.velocity[i] + c1 * random.random() * cognitive + c2 * random.random() * social

    def position_update(self):
        updated_position = ""
        for i in range(len(self.velocity)):
            updated_position += chr((ord(self.position[i]) + int(self.velocity[i])) % 256)

        self.position = updated_position


class GeneticAlgorithm:

    def init_population(self, population, buffer):

        tsize = len(GA_TARGET)

        for i in range(GA_POPSIZE):
            citizen = GA_struct("", 0)


            for j in range(tsize):
                citizen.string += chr(random.randrange(0, 90) + 32)

            population[i] = citizen

        self.population = population
        # resize(buffer, GA_POPSIZE) # ???????
        # buffer.resize(GA_POPSIZE)
        return


    def calc_fitness(self, genome=None):
        target = GA_TARGET
        tsize = len(target)

        for i in range(GA_POPSIZE):

            fitness = 0

            for j in range(tsize):
                if genome:
                    fitness = fitness + abs(ord(genome[j]) - ord(target[j]))

                else:
                    fitness = fitness + abs(ord(self.population[i].string[j]) - ord(target[j]))

            if genome:
                return fitness

            self.population[i].fitness = fitness

        return

    def sort_by_fitness(self):
        self.population.sort(key=lambda x: x.fitness-x.age)
        return

    def elitism(self, population: list[GA_struct], buffer: list[GA_struct], esize):
        # todo: check
        # buffer[:esize] = population[:esize]
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

    def mate(self, population, buffer, type, selection_method = None):

        esize = int(GA_POPSIZE * GA_ELITRATE)
        tsize = len(GA_TARGET)
        self.elitism(population, buffer, esize)


        # mate the rest
        for i in range(esize, GA_POPSIZE):

            i1 = random.randrange(0, GA_POPSIZE // 2)
            i2 = random.randrange(0, GA_POPSIZE // 2)

            if selection_method == "sus":
                parents = self.sus(population, 2)
                i1, i2 = parents[0], parents[1]

            if selection_method == "rws":
                # self.buffer = problem.RWS(problem, buffer, GA_POPSIZE)
                i1 = self.rws(population)
                i2 = self.rws(population)

                # i1 = self.roulette_selection(fitness)
                # i2 = self.roulette_selection(fitness)

            if selection_method == "tournament":
                k = 100
                i1, i2 = self.tournament_selection(population, k)

            if type == "SINGLE":
                pos = random.randrange(0, tsize)
                if selection_method != None:
                    buffer[i] = GA_struct(i1.string[0: pos] + i2.string[pos:], 0)
                else:
                    buffer[i] = GA_struct(population[i1].string[0: pos] + population[i2].string[pos:], 0)


            elif type == "DOUBLE":
                pos1 = random.randrange(0, tsize - 2)
                pos2 = random.randrange(pos1 + 1, tsize - 1)
                if selection_method != None:
                    buffer[i] = GA_struct(i1.string[0: pos1] + i2.string[pos1:pos2] + i1.string[pos2:], 0)
                else:
                    buffer[i] = GA_struct(population[i1].string[0: pos1] + population[i2].string[pos1:pos2] + population[i1].string[pos2:], 0)

            elif type == "UNIFORM":
                gen = ""
                for j in range(tsize):
                    r = random.randrange(0, 2)
                    if r == 0:
                        if selection_method != None:
                            gen = gen + i1.string[j]
                        else:
                            gen = gen + population[i1].string[j]

                    else:
                        if selection_method != None:
                            gen = gen + i2.string[j]
                        else:
                            gen = gen + population[i2].string[j]
                buffer[i] = GA_struct(gen, 0)

            if random.randrange(sys.maxsize) < GA_MUTATION:
                self.mutate(buffer[i])

        return buffer

    def print_best(self):
        print("Best: " + self.population[0].string + " fitness: " + " (" + str(self.population[0].fitness) + ")")
        return

    def swap(self, population: list[GA_struct], buffer: list[GA_struct]):

        return buffer, population

    def calcAVG(self):

        sum = 0

        for i in range(len(self.population)):
            sum += self.population[i].fitness

        return sum / GA_POPSIZE

    def calcStd(self):

        fitness = []

        for i in range(len(self.population)):
            fitness.append(self.population[i].fitness)

        return statistics.stdev(fitness)

    def BulPgia(self, population):
        for i in population:
            fitness = 0
            for j in range(len(GA_TARGET)):
                if i.string[j] == GA_TARGET[j]:  # bul pgia(letter is correct+in correct place)
                    fitness += 0
                elif i.string[j] in GA_TARGET:  # not bul pgia but letter is correct
                    fitness += 25
                else:  # not the same letter
                    fitness += 65

            i.fitness = fitness

    def rws(self, population):
        # prob = []
        # fit_sum = 0
        # for genome in population:
        #     fit_sum += genome.fitness
        #
        # for genome in population:
        #     prob.append(1 - (genome.fitness/fit_sum))
        #
        # cum_prob = cumsum(prob)
        # r = random.random()
        # for i in range(len(cum_prob) - 1):
        #     print(cum_prob)
        #     if cum_prob[i] <= (r * fit_sum):
        #         if cum_prob[i + 1] > (r * cum_prob[int(r)]):
        #             return population[i + 1].string

        # Computes the totallity of the population fitness
        population_fitness = sum([chromosome.fitness for chromosome in population])

        # Computes for each chromosome the probability
        chromosome_probabilities = [chromosome.fitness / population_fitness for chromosome in population]

        # Making the probabilities for a minimization problem
        chromosome_probabilities = (1 - np.array(chromosome_probabilities)) / (len(population) - 1)

        # Selects one chromosome based on the computed probabilities
        return np.random.choice(population, p = chromosome_probabilities)

    def roulette_selection(self, fitnesses):
        # sort the weights in ascending order
        sorted_indexed_weights = sorted(enumerate(fitnesses), key=operator.itemgetter(1))
        indices, sorted_weights = zip(*sorted_indexed_weights)
        # calculate the cumulative probability
        total_sum = sum(sorted_weights)
        probability = []
        for weight in sorted_weights:
            probability.append(weight/total_sum)
        cumulative_p = np.cumsum(probability)

        # select a random a number in the range [0,1]
        random_num = np.random.uniform(low=0, high=1)

        for index_value, cum_prob_value in zip(indices, cumulative_p):
            if random_num < cum_prob_value:
                return index_value

    def RWS(self, population, buffer, size):
        """
        Roulette wheel selection
        """
        selections = []
        fitnesses = []

        for genome in population.population:
            fitness = (1 / genome.fitness)
            fitnesses.append(fitness)

        for i in range(size):
            index = population.roulette_selection(fitnesses)
            selections.append(population.population[index])
        print("in RWS")

        return selections + buffer[size:]

    def get_subset_sum(self, population, index):
        sum = 0
        for i in range(index):
            sum += population[i].fitness
        return sum

    def sus(self, population, N):  # mn el internet
        sum = 0
        for i in range(len(population)):
            sum += 1 / population[i].fitness if population[i].fitness else 0
        point_distance = sum / N
        start_point = random.uniform(0, point_distance)
        points = [start_point + i * point_distance for i in range(N)]
        parents = set()
        while len(parents) < N:
            random.shuffle(population)  # mfhmtsh lshu b7aji?
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

        sample.sort(key=lambda x: x.fitness-x.age)
        return sample[0], sample[1]

    # def tournementSelection(self, population, buffer, size):
    #     Selections = []  # array of selections
    #     populationSize = len(population)
    #     for i in range(size):
    #         first = population[random.randrange(0, populationSize - 1)]  # pick first genom
    #         second = population[random.randrange(0, populationSize - 1)]  # pick second genom
    #
    #         if first.fitness < second.fitness:
    #             Selections.append(GA_struct(first.string, first.fitness))
    #         else:
    #             Selections.append(GA_struct(second.string, second.fitness))
    #
    #     return Selections + [i for i in buffer[size:]]

    @staticmethod
    def positive_random(rng, conflicts, filter):
        return random.choice([i for i in range(rng) if filter(conflicts[i])])

    def inversion_mutation(self):  # 7sb lkovets elle shay 7tu mfrod kmn n3'yr m7l lblock elle mnnn2e bs bl internet m7tot bs nsawe hepo5..
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
        while i < index2 + 1:  # afkr fe tre2a aktr y3ela
            self[i] = helper[j]
            j += 1
        return self

    def pso(self):
        start1 = time.time()  # clock ticks
        start2 = timeit.default_timer()  # elapsed time

        self.calc_fitness()
        particles = [] * GA_POPSIZE
        found = False

        # initializing the particles
        for item in self.population:
            velocity = list(np.random.uniform(low=0, high=1, size=len(item.string)))
            particles.append(Particle(item.string, velocity))

        global_best = self.population[0].string
        # Do while end condition or we've reached the max number of iterations
        for i in range(GA_MAXITER):
            for particle in particles:
                objective = self.calc_fitness(particle.position)  # we calculate the objective of the particle
                if objective == 0:  # this means we've found the optimal solution
                    global_best = particle.position
                    found = True
                    break
                else:
                    curr_fitness = self.calc_fitness(global_best)
                    # update the global best if we've found a better solution than the global best
                    if objective < curr_fitness:
                        global_best = particle.position

                    curr_fitness = self.calc_fitness(particle.personal_best)
                    # update the personal best if we've found a better solution than the personal best of this particle
                    if objective < curr_fitness:
                        particle.personal_best = particle.position

            if found:
                break

            # update the inertia weight
            w = (0.6 * (i - GA_MAXITER) / (GA_MAXITER ** 2)) + 0.6
            c1 = -2 * i / GA_MAXITER + 2.5
            c2 = 2 * i / GA_MAXITER + 0.5

            # for each particle, update velocity, and update position
            for particle in particles:
                particle.velocity_update(w, c1, c2, global_best)
                particle.position_update()

        elapsed = timeit.default_timer() - start2
        clock_ticks = time.time() - start1

        print("Best PSO: " + global_best + " (" + str(min(objective, curr_fitness)) + ")")
        print("Overall PSO runtime: " + str(elapsed) + " Ticks: " + str(clock_ticks))
        return global_best


if __name__ == "__main__":

    problem = GeneticAlgorithm()
    random.seed()  # whats this
    pop_alpha = [None] * GA_POPSIZE
    pop_beta = [None] * GA_POPSIZE

    problem.init_population(pop_alpha, pop_beta)


    population = pop_alpha
    buffer = pop_beta
    start_t = time.time()

    pso = problem.pso()

    for i in range(GA_MAXITER):

        time2 = time.time()  # clock ticks

        problem.BulPgia(population)
        problem.sort_by_fitness()
        problem.print_best()
        print("mean of generation is: " + str(problem.calcAVG()))
        print("standard deviation of generation is: " + str(problem.calcStd()))

        clock_ticks = time.time() - time2
        E_T = time.time() - start_t
        print("Clock ticks: " + str(clock_ticks))
        print("Elapsed time: " + str(E_T))

        fitness = []
        for j in range(len(population)):
            fitness.append(population[j].fitness)

        plt.xlabel('Fitness')
        plt.ylabel('Number of Genomes')
        plt.hist(fitness)
        # plt.show()

        if population[0].fitness == 0:
            break

        buffer = problem.mate(population, buffer, "SINGLE", "sus")
        population, buffer = problem.swap(population, buffer)

        for genome in population:
            genome.age += 1

    E_T = time.time() - start_t
    clock_ticks = time.time() - time2

    print("Elapsed time: " + str(E_T) + " Clock Ticks: " + str(clock_ticks))

