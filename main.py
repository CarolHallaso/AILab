import sys
import random
import statistics
import time
import timeit

from matplotlib import pyplot as plt
import numpy as np

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

# class Particle:
#
#     # This class represents a particle in the Particle Swarm Optimization algorithm
#
#     def __init__(self, pos):
#         self.position = pos
#         self.velocity = [random.random() for i in range(len(pos))]
#         self.personal_best = pos
#
#     def update_velocity(self, global_best, c1, c2, w):
#
#         # simply update the new velocity using the formula that we learned in the lecture
#
#         for i in range(len(self.position)):
#             first_component = c1 * random.random() * (ord(self.personal_best[i]) - ord(self.position[i]))
#             second_component = c2 * random.random() * (ord(global_best[i]) - ord(self.position[i]))
#             self.velocity[i] = self.velocity[i] * w + first_component + second_component
#
#     def update_position(self):
#
#         # simply update the new position by adding velocity
#
#         new_pos = ""
#         for i in range(len(self.position)):
#             new_pos += chr((ord(self.position[i]) + int(self.velocity[i])) % 256)
#         self.position = new_pos
class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.personal_best = position
        self.velocity = velocity

    def velocity_update(self, w, c1, c2, global_best):

        for i in range(len(self.position)):
            first_component = c1 * random.random() * (ord(self.personal_best[i]) - ord(self.position[i]))
            second_component = c2 * random.random() * (ord(global_best[i]) - ord(self.position[i]))
            self.velocity[i] = self.velocity[i] * w + first_component + second_component

        # for i in range(len(self.velocity)):
        #     cognitive = ord(self.personal_best[i]) - ord(self.position[i])
        #     social = ord(global_best[i]) - ord(self.position[i])
        #     self.velocity[i] = w * self.velocity[i] + c1 * random.random() * cognitive + c2 * random.random() * social

    def position_update(self):

        new_pos = ""
        for i in range(len(self.position)):
            new_pos += chr((ord(self.position[i]) + int(self.velocity[i])) % 256)

        self.position = new_pos


        # updated_position = ""
        # for i in range(len(self.velocity)):
        #     updated_position += chr((ord(self.position[i]) + int(self.velocity[i])) % 256)
        #
        # self.position = updated_position


class GeneticAlgorithm:

    def init_population(self, population: list, buffer: list):

        tsize = len(GA_TARGET)

        for i in range(GA_POPSIZE):
            citizen = GA_struct("", 0)

            for j in range(tsize):
                citizen.string += chr(random.randrange(0, 90) + 32)

            population[i] = citizen

        self.population = population
        #buffer.resize(GA_POPSIZE);
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
        self.population.sort(key=lambda x: x.fitness)
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

    def print_best(self):
        print("Best: " + self.population[0].string + " fitness: " + " (" + str( self.population[0].fitness) + ")")
        return

    def swap(self, population: list[GA_struct], buffer: list[GA_struct]):

        return buffer, population

    def calcAVG(self):

        sum = 0

        for i in range(len(self.population)):
            sum += self.population[i].fitness

        return sum/GA_POPSIZE

    def calcStd(self):

        fitness = []

        for i in range(len(self.population)):
            fitness.append(self.population[i].fitness)

        return statistics.stdev(fitness)

    def BulPgia(self, population):
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

    # def PSO(self):
    #
    #     particles = [] * GA_POPSIZE
    #     found = False
    #
    #     self.calc_fitness()
    #
    #     # initializing the particles
    #     for item in self.population:
    #         velocity = list(np.random.uniform(low=0, high=1, size=len(item.string)))
    #         particles.append(Particle(item.string, item.fitness, velocity))
    #
    #     global_best = self.population[0].string
    #
    #     for j in range(GA_MAXITER):
    #         for i in range(GA_POPSIZE):
    #             objective = particles[i].fitness
    #             if objective == 0:
    #                 global_best = particles[i].position
    #                 found = True
    #                 break
    #             else:
    #                 curr_fitness = self.calc_fitness(global_best)
    #                 if objective < curr_fitness:
    #                     global_best = particles[i].position
    #
    #                 if objective < particles[i].fitness:
    #                     particles[i].personal_best = particles[i].position
    #
    #                 # if objective < particles[i].fitness:
    #                 #     particles[i].personal_best = particles[i].position
    #                 # if objective < global_best:
    #                 #     global_best = objective
    #
    #                 w = 0.5 * (GA_MAXITER - i) / GA_MAXITER + 0.4
    #                 c1 = -2 * i / GA_MAXITER + 2.5
    #                 c2 = 2 * i / GA_MAXITER + 0.5
    #                 for particle in particles:
    #                     particle.velocity_update(w, c1, c2, global_best)
    #
    #                     particle.position_update()
    #
    #         if found:
    #             break
    #
    #     return global_best
        import random




    def pso(self):
        start1 = time.time()
        start2 = timeit.default_timer()

        particles = [] * GA_POPSIZE
        found = False

        self.calc_fitness()

        #initializing the particles
        for item in population:
            velocity = list(np.random.uniform(low=0, high=1, size=len(item.string)))
            particles.append(Particle(item.string, velocity))

        global_best = self.population[0].string



        for i in range(GA_MAXITER):
            for particle in particles:
                objective = self.calc_f(particle.position)
                if objective == 0:
                    global_best = particle.position
                    found = True
                    break


                curr_fitness = self.calc_f(particle.personal_best)
                if objective < curr_fitness:
                    particle.personal_best = particle.position

                best_fitness = self.calc_f(global_best)
                if objective < best_fitness:
                    global_best = particle.position

            if found:
                break
            w = 0.5 * (GA_MAXITER - i) / GA_MAXITER + 0.4
            c1 = -2 * i / GA_MAXITER + 2.5
            c2 = 2 * i / GA_MAXITER + 0.5

            for item in particles:
                item.velocity_update(w, c1, c2, global_best)
                item.position_update()




        elapsed = timeit.default_timer() - start2
        clock_ticks = time.time() - start1
        print("BestPSOOOO: " + global_best + " (" + str(min(objective, best_fitness)) + ")")
        print("Overall PSO runtime: " + str(elapsed) + " Ticks: " + str(clock_ticks))
        return global_best

    def calc_f(self, position):
        target = GA_TARGET
        tsize = len(target)

        for i in range(GA_POPSIZE):

            fitness = 0

            for j in range(tsize):
                fitness = fitness + abs(ord(position[j]) - ord(target[j]))


        return fitness



if __name__ == "__main__":

    problem = GeneticAlgorithm()
    random.seed()
    pop_alpha = [None] * GA_POPSIZE
    pop_beta = [None] * GA_POPSIZE
    problem.init_population(pop_alpha, pop_beta)
    population = pop_alpha
    buffer = pop_beta
    start_t = time.time()
    pso = problem.pso()
    print("PSO ISSSSSSSSS ")
    print(pso)
    for i in range(GA_MAXITER):

        time2 = time.time()  # clock ticks
        problem.calc_fitness()
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

        problem.mate(population, buffer, "UNIFORM")
        population, buffer = problem.swap(population, buffer)

    E_T = time.time() - start_t
    clock_ticks = time.time() - time2

    print("Elapsed time: " + str(E_T) + " Clock Ticks: " + str(clock_ticks))