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
QUEENS = 8

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

    def init_NQUEENS(self, population, buffer):
        for i in range(GA_POPSIZE):
            randArray = [random.randint(0, QUEENS - 1) for i in range(QUEENS)]  # random numbers array
            population[i] = (GA_struct(randArray, 0))
            buffer.append(GA_struct("", 0))
        self.population = population

        # ''' initialize population '''
        # for i in range(populationSize):
        #     randArray = [randint(0, numOfQueens - 1) for i in range(numOfQueens)]  # random numbers array
        #
        #     population.append(Agent(randArray, 0))
        #     buffer.append(Agent("", 0))


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

    def getFitness(self, population):
        ''' num of queens that collide'''
        for agent in population:
            count = [0 for i in range(QUEENS)]
            badCount = 0
            for j in agent.string:  # count row collisions
                count[int(j)] += 1
            for x in count:
                if x != 1:  # if collide
                    badCount += 1
            for i in range(QUEENS):
                for j in range(QUEENS):
                    if i == j:
                        continue
                    if abs(i - j) == abs(int(agent.string[i]) - int(agent.string[j])):
                        badCount += 1

            agent.fitness = badCount

    def sort_by_fitness(self):
        self.population.sort(key=lambda x: x.fitness)
        return

    def elitism(self, population: list[GA_struct], buffer: list[GA_struct], esize):
        # todo: check
        # buffer[:esize] = population[:esize]
        temp = population[:esize].copy()
        buffer[:esize] = temp
        return

    def mutate(self, member: GA_struct, problem_type):
        if problem_type == 0:
            t_size = len(member.string)
            ipos = random.randrange(0, t_size - 1)
            delta = random.randrange(0, 90) + 32
            string = member.string[: ipos] + chr((ord(member.string[ipos]) + delta) % 122) + member.string[ipos + 1:]
        elif problem_type == 1:
            t_size = QUEENS
            ipos = random.randrange(0, t_size - 1)
            delta = random.randint(0, QUEENS-1)
            string = member.string[: ipos] + [(member.string[ipos] + delta) % QUEENS] + member.string[ipos + 1:]

        member.string = string
        return

    def mate(self, population, buffer, type, method, problem_type):

        esize = int(GA_POPSIZE * GA_ELITRATE)
        if problem_type == 0:
            tsize = len(GA_TARGET)
        elif problem_type == 1:
            tsize = QUEENS

        self.elitism(population, buffer, esize)

        self.buffer = self.tournament_selection(population, buffer, esize)
        # self.buffer = problem.RWS(problem, buffer, GA_POPSIZE)

        # buffer = crossOverFunc(population, buffer, esize, len(targetString))

        # mate the rest
        for i in range(esize, GA_POPSIZE):

            i1 = random.randrange(0, GA_POPSIZE // 2)
            i2 = random.randrange(0, GA_POPSIZE // 2)

            if type == "SINGLE":
                pos = random.randrange(0, tsize)
                buffer[i] = GA_struct(population[i1].string[0: pos] + population[i2].string[pos:], 0)

            elif type == "DOUBLE":
                pos1 = random.randrange(0, tsize - 2)
                pos2 = random.randrange(pos1 + 1, tsize - 1)
                buffer[i] = GA_struct(
                    population[i1].string[0: pos1] + population[i2].string[pos1:pos2] + population[i1].string[pos2:], 0)

            elif type == "UNIFORM":
                gen = ""
                for j in range(tsize):
                    r = random.randrange(0, 2)
                    if r == 0:
                        gen = gen + population[i1].string[j]

                    else:
                        gen = gen + population[i2].string[j]
                buffer[i] = GA_struct(gen, 0)

            if random.randrange(sys.maxsize) < GA_MUTATION:
                self.mutate(buffer[i], problem_type)

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

    def BulPgia(self, population, problem_type):
        if problem_type == 0:
            length = len(GA_TARGET)
        elif problem_type == 1:
            length = QUEENS
        for i in population:
            fitness = 0
            for j in range(length):
                if i.string[j] == GA_TARGET[j]:  # bul pgia(letter is correct+in correct place)
                    fitness += 0
                elif i.string[j] in GA_TARGET:  # not bul pgia but letter is correct
                    fitness += 25
                else:  # not the same letter
                    fitness += 65

            i.fitness = fitness

    def rws(self, population):
        prob = []
        fit_sum = 0
        for genome in population:
            fit_sum += genome.fitness

        for genome in population:
            prob.append(1 - (genome.fitness/fit_sum))

        cum_prob = cumsum(prob)
        r = random.random()
        for i in range(len(cum_prob) - 1):
            print(cum_prob)
            if cum_prob[i] <= (r * fit_sum):
                if cum_prob[i + 1] > (r * cum_prob[int(r)]):
                    return population[i + 1].string


    def roulette_selection(self, fitnesses):
        # sort the weights in ascending order
        sorted_indexed_weights = sorted(enumerate(fitnesses), key=operator.itemgetter(1))
        indices, sorted_weights = zip(*sorted_indexed_weights)
        # calculate the cumulative probability
        tot_sum = sum(sorted_weights)
        prob = [x / tot_sum for x in sorted_weights]
        cum_prob = np.cumsum(prob)

        # select a random a number in the range [0,1]
        random_num = random.random()

        for index_value, cum_prob_value in zip(indices, cum_prob):
            if random_num < cum_prob_value:
                return index_value

    def RWS(self, population, buffer, size):
        '''Roulette wheel selection'''
        selections = []
        fitnesses = [(1 / GA_struct.fitness)for GA_struct in population.population]
        for i in range(size):
            index = population.roulette_selection(fitnesses)
            selections.append(population.population[index])
        print("in RWS")
        return selections + [i for i in buffer[size:]]

    def get_subset_sum(population, index):
        sum = 0
        for i in range(index):
            sum += population[i].fitness
        return sum

    def sus(population, N):  # mn el internet
        sum = 0
        for i in range(len(population)):
            sum += 1 / population[i].fitness
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
                    if population.get_subset_sum(population, j) < points[i]:
                        parents.add(population[j])
                        break
                    j += 1
                i += 1

        return list(parents)

    def tournament_selection(self, population, buffer, k):
        selections = []
        for i in range(k):
            member1 = population[random.randrange(0, len(population) - 1)]
            member2 = population[random.randrange(0, len(population) - 1)]
            if member1.fitness < member2.fitness:
                selections.append(member1)
            else:
                selections.append(member2) # why adds all K / if k is parents size then k=2 and why is it only 2??
        print("in tournament")
        return selections + [i for i in buffer[k:]] # what is the second part for??!

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

    def aging(population, buffer, size):
        Best = 5
        # we want to make the individual more fit with age
        for p in population:
            p.fitness += (p.age - Best) ** 2  # bs msh lfitness a7sn lma bser h2l wmsh aktr..!?
        return population.RWS(population, buffer, size)

    def miniConf(soln, nr, iters=1000):
        def posRand(li, filt):
            return random.choice([i for i in range(nr) if filt(li[i])])

        for k in range(iters):
            confs = soln.find_conflicts(soln, nr)
            if sum(confs) == 0:
                return soln
            col = posRand(confs, lambda elt: elt > 0)
            vconfs = [soln.hits(soln, nr, col, row) for row in range(nr)]
            soln[col] = posRand(vconfs, lambda elt: elt == min(vconfs))

    def find_conflicts(soln, nr):
        return [soln.hits(soln, nr, col, soln[col]) for col in range(nr)]

    def hits(soln, nr, col, row):
        total = 0
        for i in range(nr):
            if i == col:
                continue
            if soln[i] == row or abs(i - col) == abs(soln[i] - row):
                total += 1
        return total

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
    value = 0
    # value = input("Press 0 if you want the Genome problem, press 1 if you want the N-queens problem ")
    # print(value)
    # if value == 0:
    #     problem.init_population(pop_alpha, pop_beta)
    # elif value == 1:
    #     problem.init_NQUEENS(pop_alpha, pop_beta)

    problem.init_population(pop_alpha, pop_beta)

    # problem.init_NQUEENS(pop_alpha, pop_beta)

    population = pop_alpha
    buffer = pop_beta
    start_t = time.time()
    # problem.calc_fitness()
    # fitness = []
    # for item in population:
    #     fitness.append(item.fitness)
    # rws = problem.RWS(problem, buffer, GA_POPSIZE)

    pso = problem.pso()  # outside big loop??!

    for i in range(GA_MAXITER):

        time2 = time.time()  # clock ticks
        # if value == 0:
        #     # problem.calc_fitness()
        #     problem.BulPgia(population, value)
        # elif value == 1:
        #     problem.getFitness(population)
        problem.BulPgia(population, value)
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

        # problem.mate(population, buffer, "UNIFORM")
        buffer = problem.mate(population, buffer, "UNIFORM", problem.tournament_selection, value)
        population, buffer = problem.swap(population, buffer)

        for genome in population:
            genome.age += 1

    E_T = time.time() - start_t
    clock_ticks = time.time() - time2

    print("Elapsed time: " + str(E_T) + " Clock Ticks: " + str(clock_ticks))


















