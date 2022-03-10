import sys
import random
import statistics
import time
import timeit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


import numpy

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

'''
class ParticleSwarmOptimization:

    def f(x, y):
        return (x - 3.14) * 2 + (y - 2.72) * 2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73)

    # Compute and plot the function in 3D within [0,5]x[0,5]
    x, y = np.array(np.meshgrid(np.linspace(0, 5, 100), np.linspace(0, 5, 100)))
    z = f(x, y)

    # Find the global minimum
    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]

    # Hyper-parameter of the algorithm
    c1 = c2 = 0.1
    w = 0.8

    # Create particles
    n_particles = 20
    np.random.seed(100)
    X = np.random.rand(2, n_particles) * 5
    V = np.random.randn(2, n_particles) * 0.1

    # Initialize data
    pbest = X
    pbest_obj = f(X[0], X[1])
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()

    def update(self):
        "Function to do one iteration of particle swarm optimization"
        global V, X, pbest, pbest_obj, gbest, gbest_obj
        # Update params
        r1, r2 = np.random.rand(2)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest.reshape(-1, 1) - X)
        X = X + V
        obj = f(X[0], X[1])
        pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
        pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
        gbest = pbest[:, pbest_obj.argmin()]
        gbest_obj = pbest_obj.min()

    # Set up base figure: The contour map
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_tight_layout(True)
    img = ax.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
    fig.colorbar(img, ax=ax)
    ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
    contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
    p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.5)
    p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
    gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=100, color='black', alpha=0.4)
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])

    def animate(self, i):
        "Steps of PSO: algorithm update and show in plot"
        title = 'Iteration {:02d}'.format(i)
        # Update params
        update()
        # Set picture
        ax.set_title(title)
        pbest_plot.set_offsets(pbest.T)
        p_plot.set_offsets(X.T)
        p_arrow.set_offsets(X.T)
        p_arrow.set_UVC(V[0], V[1])
        gbest_plot.set_offsets(gbest.reshape(1, -1))
        return ax, pbest_plot, p_plot, p_arrow, gbest_plot

    anim = FuncAnimation(fig, animate, frames=list(range(1, 50)), interval=500, blit=False, repeat=True)
    anim.save("PSO.gif", dpi=120, writer="imagemagick")

    print("PSO found best solution at f({})={}".format(gbest, gbest_obj))
    print("Global optimal at f({})={}".format([x_min, y_min], f(x_min, y_min)))
'''
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

    def plotQuantiles(self, population: list[GA_struct]):

        maxf = max(pop.fitness for pop in population)    # get the max fitness for all genomes in this generation
        fitness = [0] * (maxf + 1)

        for p in population:                                  # find the histogram of genomes' fitness
            fitness[p.fitness] += 1

        tick_label = []                                         # a list that contains x-axis values
        for i in range(maxf + 1):
            tick_label.append(i)

        # plotting points as a scatter plot
        plt.scatter(tick_label, fitness, label="stars", color="green", marker=".", s=10)

        plt.xlabel('fitness')           # x-axis label
        plt.ylabel('No of genomes')     # y-axis label
        plt.title('Distribution of fitness of genomes in one generation')   # plot title
        plt.legend()
        plt.show()


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

        print("Clock ticks: " + str(clock_ticks))
        print("Elapsed time: " + str(E_T))
        #problem.plotQuantiles(population)

        if population[0].fitness == 0:
            break

        problem.mate(population, buffer, "UNIFORM")
        population, buffer = problem.swap(population, buffer)

    E_T = time.time() - start_t
    clock_ticks = time.time() - time2

    print("Elapsed time: " + str(E_T) + " Clock Ticks: " + str(clock_ticks))