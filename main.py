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

GA_POPSIZE = 2048  # genome population size
GA_MAXITER = 16384  # maximum iterations (generations)
GA_ELITRATE = 0.1  # elitism rate
GA_MUTATIONRATE = 0.25  # mutation rate
GA_MUTATION = sys.maxsize * GA_MUTATIONRATE
GA_TARGET = "Hello world!"


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
    # func that updates the particle's velocity using the inertia weights like we learned in class
    def velocity_update(self, w, c1, c2, global_best):
        for i in range(len(self.velocity)):
            cognitive = ord(self.personal_best[i]) - ord(self.position[i])
            social = ord(global_best[i]) - ord(self.position[i])
            self.velocity[i] = w * self.velocity[i] + c1 * random.random() * cognitive + c2 * random.random() * social

    # func that updates the particle's position like we learned in class
    def position_update(self):
        updated_position = ""
        for i in range(len(self.velocity)):
            updated_position += chr((ord(self.position[i]) + int(self.velocity[i])) % 256)

        self.position = updated_position


class GeneticAlgorithm:

    def init_population(self, population, buffer):
    #initialize the population
        tsize = len(GA_TARGET)

        for i in range(GA_POPSIZE):
            citizen = GA_struct("", 0)

            for j in range(tsize):
                citizen.string += chr(random.randrange(0, 90) + 32)

            population[i] = citizen

        self.population = population
        return

    def calc_fitness(self, genome=None):
        # we calculate the fitness of each genome in the population
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
        # sort the population according to the fitness in ascending order
        self.population.sort(key=lambda x: x.fitness)
        # self.population.sort(key=lambda x: x.fitness-x.age)
        return

    def elitism(self, population: list[GA_struct], buffer: list[GA_struct], esize):
        temp = population[:esize].copy()
        buffer[:esize] = temp
        return

    def mutate(self, member: GA_struct):
        # to mutate we choose a position randomly and change it to a random character
        t_size = len(member.string)
        ipos = random.randrange(0, t_size - 1)
        delta = random.randrange(0, 90) + 32
        string = member.string[: ipos] + chr((ord(member.string[ipos]) + delta) % 122) + member.string[ipos + 1:]
        member.string = string
        return

    def mate(self, population, buffer, cross_over_type, selection_method = None, probabilities = None, replacement=None, mutation_type=None):

        esize = int(GA_POPSIZE * GA_ELITRATE)
        tsize = len(GA_TARGET)
        self.elitism(population, buffer, esize)
        # we choose the parents depending on the method specified by the given input "selection_method"
        # and to each pair of parents we do a cross over according to the given input "cross_over_type"
        parents_idx = 0
        if selection_method == "sus":  # we choose the parents according to the SUS algo we learned
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

            if replacement == "PMX":
                if selection_method != None:
                    parents = self.PMX(i1.string, i2.string)
                else:
                    parents = self.PMX(population[i1].string, population[i2].string)
                i1, i2 = GA_struct(parents[0], 0), GA_struct(parents[1], 0)

            if replacement == "CX":
                if selection_method != None:
                    parents = self.CX(i1.string, i2.string)
                else:
                    parents = self.Cx(population[i1].string, population[i2].string)
                i1, i2 = GA_struct(parents[0], 0), GA_struct(parents[1], 0)

            if cross_over_type == "SINGLE":
                pos = random.randrange(0, tsize)
                if selection_method != None:
                    buffer[i] = GA_struct(i1.string[0: pos] + i2.string[pos:], 0)
                else:
                    buffer[i] = GA_struct(population[i1].string[0: pos] + population[i2].string[pos:], 0)

            elif cross_over_type == "DOUBLE":
                pos1 = random.randrange(0, tsize - 2)
                pos2 = random.randrange(pos1 + 1, tsize - 1)
                if selection_method != None:
                    buffer[i] = GA_struct(i1.string[0: pos1] + i2.string[pos1:pos2] + i1.string[pos2:], 0)
                else:
                    buffer[i] = GA_struct(population[i1].string[0: pos1] + population[i2].string[pos1:pos2] + population[i1].string[pos2:], 0)

            elif cross_over_type == "UNIFORM":
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
                if mutation_type == "inverse_mutation":
                    self.inverse_mutation(buffer[i])
                elif mutation_type == "scramble_mutation":
                    self.scramble_mutation(buffer[i])
                else:
                    self.mutate(buffer[i])

        return buffer

    def print_best(self):
        print("Best: " + self.population[0].string + " fitness: " + " (" + str(self.population[0].fitness) + ")")
        return

    def swap(self, population: list[GA_struct], buffer: list[GA_struct]):

        return buffer, population

    def calcAVG(self):
        # calculates and returns the average fitness of the population
        sum = 0
        for i in range(len(self.population)):
            sum += self.population[i].fitness

        return sum / GA_POPSIZE

    def calcStd(self):
        # calculates and returns the STD of the population
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

    def inverse_mutation(self, member):
        # implements the inversion mutation we learned
        p = member.string
        index1 = random.randrange(0, len(p))
        index2 = random.randrange(index1, len(p))
        subs1 = p[0:index1]
        subs2 = p[index1:index2]
        subs3 = p[index2:]
        tmp = subs2[len(subs2)::-1]
        txt = [subs1, tmp, subs3]
        p = "".join(txt)
        member.string = p

    def scramble_mutation(self, member):
        # implements the scramble mutation we learned
        p = member.string
        index1 = random.randrange(0, len(p))
        index2 = random.randrange(index1, len(p))
        subs1 = p[0:index1]
        subs2 = p[index1:index2]
        subs3 = p[index2:]
        tmp = ''.join(random.sample(subs2, len(subs2)))
        txt = [subs1, tmp, subs3]
        p = "".join(txt)
        member.string = p

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

    def init_roulette(self, population):
        # build the roulette with the probabilities being according to the fitnesses
        problem.calc_fitness()
        probs = []
        total_fitness = 0
        for i in range(GA_POPSIZE):
            total_fitness += population[i].fitness
        for i in range(GA_POPSIZE):
            probs.append(population[i].fitness / total_fitness)
        return probs

    def PMX(self, p1, p2):
        # implements Partially Matched crossover
        r = random.randrange(0, len(p1))
        tmp1 = p1[r]
        tmp2 = p2[r]
        p1 = p1.replace(tmp1, tmp2)
        p2 = p2.replace(tmp2, tmp1)

        return p1, p2

    def search_for_num(self, num):
        for i in range(len(self)):
            if self[i] == num:
                return i

    def CX(self, p1, p2):
        # implements Cycle crossover
        parents = set()
        p1new = p1
        p2new = p2
        indx = 0
        next_index = 0
        t = 0
        sum = 0
        while next_index <= (len(p1)):
            start = p1[next_index]
            if p1[indx] == -1:
                break
            next_index += 1
            num = p2[indx]
            while num != start:
                indx = p1.find(num)
                num = p2[indx]
                if indx == next_index:
                    next_index += 1
                if t % 2 == 0:
                    subs1 = p1new[0:indx]
                    subs2 = p1new[indx + 1:]
                    txt = [subs1, p1[indx], subs2]
                    p1new = "".join(txt)
                    subk1 = p2new[0:indx]
                    subk2 = p2new[indx + 1:]
                    txt = [subk1, p2[indx], subk2]
                    p2new = "".join(txt)
                else:
                    subs1 = p1new[0:indx]
                    subs2 = p1new[indx + 1:]
                    txt = [subs1, p2[indx], subs2]
                    p1new = "".join(txt)
                    subk1 = p2new[0:indx]
                    subk2 = p2new[indx + 1:]
                    txt = [subk1, p1[indx], subk2]
                    p2new = "".join(txt)

                sum += 1

            t += 1
        parents.add(p1new)
        parents.add(p2new)
        return parents
        # parents = set()
        # p2new = None
        # p1new = None
        # indx = 0
        # next_index = 0
        # t = 0
        # sum = 0
        # while next_index <= (len(p1)):
        #     start = p1[next_index]
        #     if p1[indx] == -1:
        #         break
        #     next_index += 1
        #     num = p2[indx]
        #     while num != start:
        #         indx = p1.search_for_num(p1, num)
        #         num = p2[indx]
        #         if indx == next_index:
        #             next_index += 1
        #         if t % 2 == 0:
        #             p1new[indx] = p1[indx]
        #             p2new[indx] = p2[indx]
        #         else:
        #             p1new[indx] = p2[indx]
        #             p2new[indx] = p1[indx]
        #         p1[indx] = -1
        #         p2[indx] = -1
        #         sum += 1
        #
        #     t += 1
        # parents.add(p1new)
        # parents.add(p2new)
        # return parents



if __name__ == "__main__":

    problem = GeneticAlgorithm()
    random.seed()
    pop_alpha = [None] * GA_POPSIZE
    pop_beta = [None] * GA_POPSIZE

    problem.init_population(pop_alpha, pop_beta)  #initialize the population

    population = pop_alpha
    buffer = pop_beta
    start_t = time.time()  # starting time

    pso = problem.pso()

    probabilities = problem.init_roulette(population)  # initialize the roulette

    for i in range(GA_MAXITER):

        time2 = time.time()  # clock ticks

        # problem.calc_fitness()
        problem.BulPgia(population)  # calculate the fitness according to bul pgia
        problem.sort_by_fitness()
        problem.print_best()
        print("mean of generation is: " + str(problem.calcAVG()))
        print("standard deviation of generation is: " + str(problem.calcStd()))

        clock_ticks = time.time() - time2
        E_T = time.time() - start_t
        print("Clock ticks: " + str(clock_ticks))
        print("Elapsed time: " + str(E_T))

        # plot histogram of the fitnesses of the population in each iteration
        # fitness = []
        # for j in range(len(population)):
        #     fitness.append(population[j].fitness)

        # plt.xlabel('Fitness')
        # plt.ylabel('Number of Genomes')
        # plt.hist(fitness)
        # plt.show()

        if population[0].fitness == 0:  # we've reached a solution
            break
        # to mate the population you need to specify what kind of cross over you want ("SINGLE", "DOUBLE, or "UNIFORM")
        # and the method to choose parents ("tournamet", "rws", "sus", or None)
        # if you choose rws you also need to give the func the probabilities list
        # and the cross over for the two parents ("PMX", "CX", or None)
        # and the mutation type ("inverse_mutation", "scramble_mutation", or None)
        buffer = problem.mate(population, buffer, "DOUBLE", "tournament", probabilities, "CX", "scramble_mutation")  # mate the population
        population, buffer = problem.swap(population, buffer)

        for genome in population:  # add the age of the genomes in every iteration
            genome.age += 1

    E_T = time.time() - start_t  # calculate end time
    clock_ticks = time.time() - time2

    print("Elapsed time: " + str(E_T) + " Clock Ticks: " + str(clock_ticks))

