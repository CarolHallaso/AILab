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

    @staticmethod
    def init_population(population: list, buffer: list):

        tsize = len(GA_TARGET)

        for i in range(GA_POPSIZE):
            citizen = GA_struct("", 0)

            for j in range(tsize):
                citizen.string += chr(random.randrange(0, 90) + 32)

            population[i] = citizen
        #buffer.resize(GA_POPSIZE);
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

        return sum/GA_POPSIZE

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

   # def PSO(self, population):
        #for i in range(GA_POPSIZE)
          #  population[i].string =
          #rando init ben unknown bound values
    def roulette_selection(weights):
        '''performs weighted selection or roulette wheel selection on a list
        and returns the index selected from the list'''

        # sort the weights in ascending order
        sorted_indexed_weights = sorted(enumerate(weights), key=operator.itemgetter(1));
        indices, sorted_weights = zip(*sorted_indexed_weights);
        # calculate the cumulative probability
        tot_sum = sum(sorted_weights)
        prob = [x / tot_sum for x in sorted_weights]
        cum_prob = np.cumsum(prob)
        # select a random a number in the range [0,1]
        random_num = random()

        for index_value, cum_prob_value in zip(indices, cum_prob):
            if random_num < cum_prob_value:
                return index_value
def RWS(population, buffer, size):
    '''Roulette wheel selection'''
    selections = []
    fit = [(1/agent.fitness) for agent in population]
    for i in range(size):
        index = roulette_selection(fit)
        selections.append(population[index])

    return selections + [i for i in buffer[size:]]

if __name__ == "__main__":

    problem = GeneticAlgorithm()
    random.seed()
    pop_alpha = [None] * GA_POPSIZE
    pop_beta = [None] * GA_POPSIZE
    problem.init_population(pop_alpha, pop_beta)
    population = pop_alpha
    buffer = pop_beta
    start_t = time.time()

    for i in range(GA_MAXITER):

        time2 = time.time()  # clock ticks
        problem.calc_fitness(population)
        problem.sort_by_fitness(population)
        problem.print_best(population)
        print("mean of generation is: " + str(problem.calcAVG(population)))
        print("standard deviation of generation is: " + str(problem.calcStd(population)))

        clock_ticks = time.time() - time2
        E_T = time.time() - start_t


        fitness = []
        for j in range(len(population)):
            fitness.append(population[j].fitness)
        
        print("Clock ticks: " + str(clock_ticks))
        print("Elapsed time: " + str(E_T))
        
        plt.xlabel('Fitness')
        plt.ylabel('Number of Genomes')
        plt.hist(fitness)
      # plt.show()


        


        if population[0].fitness == 0:
            break

        problem.mate(population, buffer, "UNIFORM")
        population, buffer = problem.swap(population, buffer)



    E_T = time.time() - start_t
    clock_ticks = time.time() - time2


    print("Elapsed time: " + str(E_T) + " Clock Ticks: " + str(clock_ticks))