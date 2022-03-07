import sys
import random
import statistics
import time
import timeit

from matplotlib import pyplot as plt

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

class GeneticAlgorithm:

    def init_population(self, population: list, buffer: list):

        tsize = len(GA_TARGET)

        for i in range(GA_POPSIZE):
            citizen = GA_struct("", 0)

            for j in range(tsize):
                citizen.string += chr(random.randrange(0, 90) + 32)

            population[i] = citizen

        return

    def calc_fitness(self, population: list[GA_struct]):

        target = GA_TARGET
        tsize = len(target)

        for i in range(GA_POPSIZE):

            fitness = 0

            for j in range(tsize):
                fitness = fitness + abs(ord(population[i].string[j]) - ord(target[j]))

            population[i].fitness = fitness

        return

    def sort_by_fitness(self, population: list[GA_struct]):
        population.sort(key=lambda x: x.fitness)
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

    def mate(self, population: list[GA_struct], buffer: list[GA_struct], type):

        esize = int(GA_POPSIZE * GA_ELITRATE)
        tsize = len(GA_TARGET)
        self.elitism(population, buffer, esize)

        # mate the rest
        for i in range(esize, GA_POPSIZE):

            i1 = random.randrange(0, GA_POPSIZE // 2)
            i2 = random.randrange(0, GA_POPSIZE // 2)

            if type == "SINGLE":
                pos = random.randrange(0, tsize)
                buffer[i] = GA_struct(population[i1].string[0: pos] + population[i2].string[pos:], 0)

            elif type == "DOUBLE":
                pos1 = random.randrange(0, tsize-2)
                pos2 = random.randrange(pos1+1, tsize-1)
                buffer[i] = GA_struct(population[i1].string[0: pos1] + population[i2].string[pos1:pos2] + population[i1].string[pos2:], 0)

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
                self.mutate(buffer[i])

        return

    def print_best(self, gav: list[GA_struct]):
        print("Best: " + gav[0].string + " fitness: " + " (" + str(gav[0].fitness) + ")")
        return

    def swap(self, population: list[GA_struct], buffer: list[GA_struct]):

        return buffer, population

    def calcAVG(self, population: list[GA_struct]):

        sum = 0

        for i in range(len(population)):
            sum += population[i].fitness

        return sum/len(population)

    def calcStd(self, population: list[GA_struct]):

        fitness = []

        for i in range(len(population)):
            fitness.append(population[i].fitness)

        return statistics.stdev(fitness)


    def BulPgia(population):
        for i in population:
            fitness = 0
            for j in range(len(GA_TARGET)):
                if i.string[j] == GA_TARGET[j]:  # if bul pgia
                    fitness += 0
                elif i.string[j] in GA_TARGET:  # if not bul pgia but letter is correct
                    fitness += 20
                else:  # if not even a same letter
                    fitness += 70

            i.fitness = fitness


if __name__ == "__main__":

    start1 = time.time()                       # to measure clock ticks
    start2 = timeit.default_timer()            # to measure elapsed time

    problem = GeneticAlgorithm()
    random.seed()
    pop_alpha = [None] * GA_POPSIZE
    pop_beta = [None] * GA_POPSIZE
    problem.init_population(pop_alpha, pop_beta)
    population = pop_alpha
    buffer = pop_beta

    for i in range(GA_MAXITER):

        generation_start = timeit.default_timer()

        problem.calc_fitness(population)
        problem.sort_by_fitness(population)
        problem.print_best(population)
        print("mean of generation is: " + str(problem.calcAVG(population)))
        print("standard deviation of generation is: " + str(problem.calcStd(population)))

        elapsed = timeit.default_timer() - generation_start
        clock_ticks = time.time() - start1

        fitness = []
        for i in range(len(population)):
            fitness.append(population[i].fitness)

        plt.xlabel('Fitness')
        plt.ylabel('Number of Genomes')
        plt.hist(fitness)
        plt.show()


        if population[0].fitness == 0:
            break

        problem.mate(population, buffer, "UNIFORM")
        population, buffer = problem.swap(population, buffer)

    elapsed = timeit.default_timer() - start2
    clock_ticks = time.time() - start1
    # clockticks 02:08 shayyyyyy
    print("Overall runtime: " + str(elapsed) + " Ticks: " + str(clock_ticks))