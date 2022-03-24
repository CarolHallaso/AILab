import math
import random
import sys
import time
import numpy as np

POPSIZE = 2048  # genome population size
MAXITER = 100  # maximum iterations
ELITRATE = 0.1  # elitism rate
MUTATIONRATE = 0.25  # mutation rate
MUTATION = sys.maxsize * MUTATIONRATE
e = 2.71828182846

class GA_struct:

    # Citizens of our Population
    def __init__(self, permutation, fitness):
        self.permutation = permutation
        self.fitness = fitness

class BinPacking:

    def __init__(self, objects, Bsize):
        self.Bsize = Bsize
        self.objects = objects
        self.NumOfObjects = len(objects)

    def init_population(self, population: list):

        for i in range(POPSIZE):
            citizen = GA_struct([], 0)
            citizen.permutation = np.random.permutation(self.NumOfObjects)
            population[i] = citizen
        return

    def calc_fitness(self, population: list):
        capacity = self.Bsize
        n = self.NumOfObjects
        k = 2
        for i in range(POPSIZE):
            num_of_bins = len(set(population[i].permutation))
            bins = [0] * n
            sum = 0

            for j in range(n):
                sum += self.objects[j]
                helper = population[i].permutation
                bins[helper[j]] = sum

            fitness = 0
            for j in range(n):
                if bins[j] != 0:
                    fitness += pow(abs(capacity - bins[j]), k)

            fitness /= num_of_bins
            population[i].fitness = fitness
        return

    def sort_by_fitness(self, population: list):
        population.sort(key=lambda x: x.fitness)
        return

    def print_best(self, population: list[GA_struct]):
        print(f"Best till now: {len(set(population[0].permutation))} bins ")
        return

    def elitism(self, population: list[GA_struct], buffer: list[GA_struct], size):
        tmp = population[:size].copy()
        buffer[:size] = tmp
        return

    def mutate(self, citizen):
        size = len(citizen.permutation)
        index = random.randrange(0, size)
        b = random.randrange(0, self.NumOfObjects)
        citizen.permutation[index] = b
        return citizen.permutation

    def mate(self, population: list[GA_struct], buffer: list[GA_struct]):
        size = int(POPSIZE * ELITRATE)
        self.elitism(population, buffer, size)
        for i in range(size, POPSIZE):
            indx1 = random.randrange(1, self.NumOfObjects)
            indx2 = random.randrange(indx1, self.NumOfObjects)
            c1 = random.randrange(0, POPSIZE // 2)
            c2 = random.randrange(0, POPSIZE // 2)

            buffer[i] = GA_struct(
                list(population[c1].permutation[0: indx1]) + list(population[c2].permutation[indx1:indx2]) + list(population[c1].permutation[indx2:]), 0)

            if random.randrange(sys.maxsize) < MUTATION:
                self.mutate(buffer[i])
        return

    def swap(self, population: list[GA_struct], buffer: list[GA_struct]):

        return buffer, population

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

    def genetic(self, objects, capacity):
        time0 = time.time()
        random.seed()
        pop1 = [None] * POPSIZE
        pop2 = [None] * POPSIZE
        self.init_population(pop1)
        population = pop1
        buffer = pop2
        generation_num = 0
        for i in range(MAXITER):
            generation_num += 1
            self.calc_fitness(population)
            self.sort_by_fitness(population)
            print("Genetic algorithm -")
            self.print_best(population)

            best_fitness = population[0].fitness
            number_of_best_genomes = self.get_num_of_genomes_with_fitness(population, best_fitness)
            mid = len(population) / 2
            mid = math.floor(mid)
            mid_fitness = population[mid].fitness
            num_of_mid_genomes = self.get_num_of_genomes_with_fitness(population, mid_fitness)
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

            #genetic_diversity = self.calculate_genetic_diversity(population)

            #print("Genetic Diversity: " + str(genetic_diversity))

            self.mate(population, buffer)
            population, buffer = self.swap(population, buffer)

            # uniform decay mutation
            # rate = GA_MUTATIONRATE * (1 / GA_MAXITER)
            # GA_MUTATIONRATE = GA_MUTATIONRATE - rate

            # Adaptive decrease function mutation
            pmax = 0.3
            r = 0.5
            helper1 = 2 * (pmax ** 2) * (e ** (r * generation_num))
            helper2 = pmax + (pmax * (e ** (r * generation_num)))
            MUTATIONRATE = helper1 / helper2

        elapsed_time = time.time() - time0
        print("Genetic Algorithm Elapsed time = " + str(elapsed_time))
        return len(set(population[0].permutation))


# Returns number of bins required using first fit algorithm
def firstFit(objects, capacity):
        time0 = time.time()  # Create an array to store remaining space in bins
        # there can be at most n bins
        bin_rem = [0] * num_of_objects

        # Initialize result (Count of bins)
        result = 0
        for i in range(num_of_objects):
            # Find the first bin that has enough space
            j = 0
            while j < result:
                if bin_rem[j] >= objects[i]:
                    bin_rem[j] = bin_rem[j] - objects[i]
                    break
                j += 1

            # If no bin could accommodate weight[i]
            if j == result:
                bin_rem[result] = capacity - objects[i]
                result = result + 1

        elapse_time = time.time()-time0
        print("First Fit Algorithm - Elapsed Time = " + str(elapse_time))
        return result

if __name__ == "__main__":

    for k in range (4):
        num_of_objects = 50
        capacity = 100
        print("For problem number " + str(k+1))
        print("Number of objects is: " + str(num_of_objects))
        print("Bins max capacity is: " + str(capacity))
        random.seed()
        objects = [0] * num_of_objects
        for i in range(num_of_objects):
            # each object weigh can be <= capacity
            objects[i] = random.randrange(1, capacity + 1)

        print("The objects we are trying to pack in this problem")
        print(objects)
        result1 = firstFit(objects, capacity)
        print("Number of bins required in - First fit Algorithm:")
        print(result1)

        problem = BinPacking(objects, capacity)
        result2 = problem.genetic(objects, capacity)
        print("Number of bins required in - Genetic Algorithm:")
        print(result2)









