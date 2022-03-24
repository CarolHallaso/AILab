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

GA_POPSIZE = 2048  # genome population size
GA_MAXITER = 16384  # maximum iterations (generations)
GA_ELITRATE = 0.1  # elitism rate
GA_MUTATIONRATE = 0.25  # mutation rate
GA_MUTATION = sys.maxsize * GA_MUTATIONRATE
GA_TARGET = "Hello world!"
e = 2.71828182846



class GA_struct:

    # Citizens of our Population
    def __init__(self, string, fitness):
        self.string = string
        self.fitness = fitness
        self.age = 0
        self.species = -1


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

    def sort_by_species(self):
        self.population.sort(key=lambda x: x.species)

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

    def get_num_of_genomes_in_species(self, population, species): #afkr fsh 7aji
        count = 0
        for i in range(len(population)):
            if population[i].species == species:
                count += 1
        return count

    def get_two_random_parents_with_same_species(self, population, species):
        count = 0
        genomes = []
        for i in range(len(population)):
            if population[i].species == species:
                genomes.append(population[i])
                count += 1
        r1 = random.randrange(0, count)
        r2 = random.randrange(0, count)
        return genomes[r1], genomes[r2]

    def get_num_of_species(self, population):
        species = []
        for i in range(len(population)):
            found = 0
            for j in range(len(species)):
                if population[i].species == species[j]:
                    found = 1
            if found == 0:
                species.append(population[i].species)
        return len(species)

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

            if selection_method == "threshold":
                #self.sort_by_species() afkr fsh 7aji
                num_of_species = self.get_num_of_species(population)
                s = random.randrange(0, num_of_species)
                num = self.get_num_of_genomes_in_species(population, s)
                i1, i2 = self.get_two_random_parents_with_same_species(population, s)

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

    def get_num_of_genomes_with_fitness(self, population, fitness):
        i = 0
        num = 0
        for i in range(len(population)):
            if population[i].fitness == fitness:
                num += 1
                i += 1

        return num

    def editDistDP(self, str1, str2, m, n):
        # Create a table to store results of subproblems
        dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

        # Fill d[][] in bottom up manner
        for i in range(m + 1):
            for j in range(n + 1):

                # If first string is empty, only option is to
                # insert all characters of second string
                if i == 0:
                    dp[i][j] = j  # Min. operations = j

                # If second string is empty, only option is to
                # remove all characters of second string
                elif j == 0:
                    dp[i][j] = i  # Min. operations = i

                # If last characters are same, ignore last char
                # and recur for remaining string
                elif str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]

                # If last character are different, consider all
                # possibilities and find minimum
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                       dp[i - 1][j],  # Remove
                                       dp[i - 1][j - 1])  # Replace

        return dp[m][n]

    def calculate_distance(self, first, second):

        diff = problem.editDistDP(first.string, second.string, len(first.string), len(second.string))

        return diff

    def calculate_genetic_diversity(self, population):
        ans = 0
        for i in range(len(population)):

            for j in range(len(population)):
                ans += problem.calculate_distance(population[i], population[j])
        return ans / len(population)

    def threshold(self, population):
        threshold = GA_POPSIZE * 0.3
        speciation = [[]] * 30
        num_of_species = 30
        for i in range(len(population)):
            check = 0
            for j in range(len(speciation)):

                for x in range(len(speciation[j])):

                    if self.calculate_distance(population[i], speciation[j][x]) <= threshold:
                        check += 1

                if check == len(speciation[j]):
                    speciation[j].append(population[i])
                    population[i].species = j

            if population[i].species == -1:
                population[i].species = num_of_species
                num_of_species += 1
                speciation.append([population[i]])




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

    generation_num = 0

    for i in range(GA_MAXITER):

        time2 = time.time()  # clock ticks

        generation_num += 1

        # problem.calc_fitness()
        problem.BulPgia(population)  # calculate the fitness according to bul pgia
        problem.sort_by_fitness()
        problem.print_best()
        print("mean of generation is: " + str(problem.calcAVG()))
        print("standard deviation of generation is: " + str(problem.calcStd()))
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
        if(prob_mid != 0):
            selection_pressure = prob_best / prob_mid


        print("selection pressure:" + str(selection_pressure))

        #genetic_diversity = problem.calculate_genetic_diversity(population)

        #print("Genetic Diversity: " + str(genetic_diversity))


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
        #problem.threshold(population) #m3 hd bsht3'l bs bser kter btee2!!
        buffer = problem.mate(population, buffer, "DOUBLE", "tournament", probabilities)  # mate the population
        population, buffer = problem.swap(population, buffer)

        for genome in population:  # add the age of the genomes in every iteration
            genome.age += 1

        # uniform decay mutation
        #rate = GA_MUTATIONRATE * (1 / GA_MAXITER)
        #GA_MUTATIONRATE = GA_MUTATIONRATE - rate

        # Adaptive decrease function mutation
        pmax = 0.3
        r = 0.5
        helper1 = 2 * (pmax ** 2) * (e ** (r * generation_num))
        helper2 = pmax + (pmax * (e ** (r*generation_num)))
        GA_MUTATIONRATE = helper1 / helper2







    E_T = time.time() - start_t  # calculate end time
    clock_ticks = time.time() - time2

    print("Elapsed time: " + str(E_T) + " Clock Ticks: " + str(clock_ticks))
