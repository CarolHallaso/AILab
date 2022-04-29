import math
import random
import time


import numpy as np
from numpy.random import randint
import sys

from matplotlib import pyplot as plt

Iterations = 1000
GA_POPSIZE = 1000
GA_ELITRATE = 0.1
GA_MUTATIONRATE = 0.25

class ReadData:

    def read_data(self, file):
        f = open(file, 'r')

        while f.readline().find('TYPE') == -1:
            pass
        line = f.readline()
        # DIMENSION : 22
        (word, points, dimension) = line.split()
        dimension = int(dimension)

        while f.readline().find('EDGE_WEIGHT_TYPE') == -1:
            pass
        line = f.readline()
        (word, points, capacity) = line.split()
        capacity = int(capacity)

        while f.readline().find('NODE_COORD_SECTION') == -1:
            pass

        nodes_coordinates = []

        line = f.readline()
        while line.find('DEMAND_SECTION') == -1:
            (word, x, y) = line.split()
            nodes_coordinates.append(([float(x), float(y)]))
            line = f.readline()

        demand = []
        line = f.readline()
        while line.find('DEPOT_SECTION') == -1:
            (word, d) = line.split()
            demand.append(float(d))
            line = f.readline()

        x = f.readline()
        y = f.readline()
        depot = [float(x), float(y)]

        return dimension, capacity, nodes_coordinates, demand, depot

class GA_struct:

    # Citizens of our Population
    def __init__(self, permutation, fitness, cars):
        self.permutation = permutation
        self.fitness = fitness
        self.cars = cars


class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.personal_best = position
        self.velocity = velocity
    # func that updates the particle's velocity using the inertia weights like we learned in class
    def velocity_update(self, w, c1, c2, global_best):
        for i in range(len(self.velocity)):
            cognitive = self.personal_best[i] - self.position[i]
            social = global_best[i] - self.position[i]
            self.velocity[i] = w * self.velocity[i] + c1 * random.random() * cognitive + c2 * random.random() * social

    # func that updates the particle's position like we learned in class
    def position_update(self):
        len = 22
        cvrp = CVRP()
        updated_position = CVRP.generate_per(cvrp, len)

        self.position = updated_position

class CVRP:

    def generate_per(self, length):
        randper = np.random.permutation(length)
        return randper

    def calculate_distance(self, point1, point2):
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        diffx = (x2 - x1) ** 2
        diffy = (y2 - y1) ** 2
        dist = math.sqrt(diffx + diffy)
        return dist

    def print_best_sol(self, permutation, cars):
        iterator = 0
        j = 0
        i = 0
        while i < (len(permutation)):
            num = cars[j]
            print(0, end=' ')
            for k in range(num):
                print(permutation[i] + 1, end=' ')
                i = i + 1
            j = j + 1
            print(0, end=' ')
            print()


    def Simulated_Annealing(self, dimension, capacity, coordsection, demand, depot):
        startt = time.time()

        best_sol = [0]
        best_cost = sys.maxsize
        cars_best_sol = []
        cars = [0] * dimension
        current_car = 0
        current_car_capacity = capacity
        temperature = 5000

        x = [0] * Iterations
        fit = [0] * Iterations

        for i in range(Iterations):
            x[i] = i
            permutation = self.generate_per(dimension)
            overall_distance = 0
            current_car = 0
            current_car_capacity = capacity
            cars = [0] * dimension

            for j in range(len(permutation)):
                distance = 0
                wayback_dist = 0
                if current_car_capacity > demand[permutation[j]]:
                    current_car_capacity = current_car_capacity - demand[permutation[j]]
                    cars[current_car] = cars[current_car] + 1
                    if j > 0:
                        # not the first node so we calculate the distance between the previous node and the current one
                        distance = self.calculate_distance(coordsection[permutation[j-1]], coordsection[permutation[j]])
                    elif j == 0:
                        # the first node so we calculate the distance between the start point and the current node
                        distance = self.calculate_distance(depot, coordsection[permutation[j]])
                else:  # we need a new car
                    wayback_dist = self.calculate_distance(coordsection[permutation[j-1]], depot)
                    current_car = current_car + 1
                    current_car_capacity = capacity
                    cars[current_car] = cars[current_car] + 1
                    distance = self.calculate_distance(depot, coordsection[permutation[j]])

                overall_distance = overall_distance + distance + wayback_dist

            if wayback_dist == 0:
                wayback_dist = self.calculate_distance(coordsection[permutation[j-1]], depot)
                overall_distance = overall_distance + wayback_dist
            if overall_distance < best_cost:
                best_cost = overall_distance
                best_sol = permutation
                cars_best_sol = cars

            elif math.exp((best_cost - overall_distance) / temperature) > random.random():
                best_cost = overall_distance
                best_sol = permutation
                cars_best_sol = cars

            temperature = temperature - 5

            fit[i] = best_cost

        print(best_sol)
        print(cars_best_sol)
        self.print_best_sol(best_sol, cars_best_sol)
        elapsed = time.time() - startt
        print("Overall SA runtime: " + str(elapsed))



        #plt.plot(x, fit, label="fitness")
        #plt.legend()
        #plt.show()

        return best_cost


    def init_pop(self, population, length):
        for i in range(GA_POPSIZE):
            citizen = GA_struct([], 0, [])
            citizen.permutation = self.generate_per(length)

            population[i] = citizen

        return population

    def calc_fitness(self, population, dimension, capacity, nodes, demand, depot):

        for i in range(len(population)):

            permutation = population[i].permutation

            overall_distance = 0
            current_car = 0
            current_car_capacity = capacity
            cars = [0] * dimension

            for j in range(len(permutation)):
                distance = 0
                wayback_dist = 0
                if current_car_capacity > demand[permutation[j]]:
                    current_car_capacity = current_car_capacity - demand[permutation[j]]
                    cars[current_car] = cars[current_car] + 1
                    if j > 0:
                        # not the first node so we calculate the distance between the previous node and the current one
                        distance = self.calculate_distance(nodes[permutation[j - 1]],
                                                           nodes[permutation[j]])
                    elif j == 0:
                        # the first node so we calculate the distance between the start point and the current node
                        distance = self.calculate_distance(depot, nodes[permutation[j]])
                else:  # we need a new car
                    wayback_dist = self.calculate_distance(nodes[permutation[j - 1]], depot)
                    current_car = current_car + 1
                    current_car_capacity = capacity
                    cars[current_car] = cars[current_car] + 1
                    distance = self.calculate_distance(depot, nodes[permutation[j]])

                overall_distance = overall_distance + distance + wayback_dist

            if wayback_dist == 0:
                wayback_dist = self.calculate_distance(nodes[permutation[j - 1]], depot)
                overall_distance = overall_distance + wayback_dist

            population[i].fitness = overall_distance
            population[i].cars = cars

        return population

    def sort_by_fitness(self, population):
        # sort the population according to the fitness in ascending order
        population.sort(key=lambda x: x.fitness)

        return population

    def elitism(self, population: list[GA_struct], buffer: list[GA_struct], esize):
        temp = population[:esize].copy()
        buffer[:esize] = temp
        return buffer

    def checkifexist(self, permutation, num):
        exist = 0
        for i in range(len(permutation)):
            if permutation[i] == num:
                exist = 1
                break
        return exist

    def find_first_not_exist(self, permutation, newp):
        for i in range(len(permutation)):
            if self.checkifexist(newp, permutation[i]) == 0:
                return permutation[i]
        return -1

    def search_for_num(self, permutation, num):
        for i in range(len(permutation)):
            if permutation[i] == num:
                return i
        return -1


    def GAcrossover(self, p1, p2):
        newp = [0] * len(p1)
        iterator = 0

        start = random.randrange(0, len(p1))
        counter = 0

        num1 = p1[start]

        while counter <= len(p1):
            exist = self.checkifexist(p1, num1)
            if exist == 0:
                newp[iterator] = num1
                counter = counter + 1
                iterator = iterator + 1
            else:
                helper = self.find_first_not_exist(p1, newp)
                if helper != -1:
                    num1 = helper
                else:
                    break
                newp[iterator] = num1
                counter = counter + 1
                iterator = iterator + 1
            i2 = self.search_for_num(p2, num1)
            if self.checkifexist(newp, p2[i2]) == 1 or i2 == 0:
                helper = self.find_first_not_exist(p2, newp)
                if helper != -1:
                    num2 = helper
                else:
                    break
            else:
                num2 = p2[i2 + 1]
            newp[iterator] = num2
            iterator = iterator + 1
            counter = counter + 1
            i1 = self.search_for_num(p1, num2)
            if self.checkifexist(newp, p1[i1]) == 1 or i1 == 0:
                helper = self.find_first_not_exist(p1, newp)
                if helper != -1:
                    num1 = helper
                else:
                    break
            else:
                num1 = p1[i1 + 1]

        return newp

    def mate(self, population, buffer):

        esize = int(GA_POPSIZE * GA_ELITRATE)
        buffer = self.elitism(population, buffer, esize)

        # mate the rest
        for i in range(esize, GA_POPSIZE):

            i1 = random.randrange(0, GA_POPSIZE // 2)
            i2 = random.randrange(0, GA_POPSIZE // 2)


            new_per = self.GAcrossover(population[i1].permutation, population[i2].permutation)
            buffer[i].permutation = new_per

        return buffer

    def swap(self, population: list[GA_struct], buffer: list[GA_struct]):

        return buffer, population

    def Genetic_Algorithm(self, dimension, capacity, coordsection, demand, depot):
        startt = time.time()

        pop_alpha = [None] * GA_POPSIZE
        pop_beta = [None] * GA_POPSIZE

        pop_alpha = self.init_pop(pop_alpha, dimension)
        buffer = self.init_pop(pop_beta, dimension)

        population = pop_alpha

        x = [0] * 100
        fit = [0] * 100

        for i in range(100):
            x[i] = i
            population = self.calc_fitness(population, dimension, capacity, coordsection, demand, depot)
            population = self.sort_by_fitness(population)

            buffer = self.mate(population, buffer)  # mate the population
            population, buffer = problem.swap(population, buffer)

            fit[i] = population[0].fitness

        self.print_best_sol(population[0].permutation, population[0].cars)

        elapsed = time.time() - startt
        print("Overall GA runtime: " + str(elapsed))

        #plt.title('GA:')
        #plt.plot(x, fit, label="fitness")
        #plt.legend()
        #plt.show()

        return population[0].permutation, population[0].fitness


    def calc_cost(self, permutation, dimension, capacity, nodes, demand, depot):
        overall_distance = 0
        current_car = 0
        current_car_capacity = capacity
        cars = [0] * dimension
        wayback_dist = 0

        for j in range(len(permutation)):
            distance = 0
            wayback_dist = 0
            if current_car_capacity > demand[permutation[j]]:
                current_car_capacity = current_car_capacity - demand[permutation[j]]
                cars[current_car] = cars[current_car] + 1
                if j > 0:
                    # not the first node so we calculate the distance between the previous node and the current one
                    distance = self.calculate_distance(nodes[permutation[j - 1]],
                                                       nodes[permutation[j]])
                elif j == 0:
                    # the first node so we calculate the distance between the start point and the current node
                    distance = self.calculate_distance(depot, nodes[permutation[j]])
            else:  # we need a new car
                wayback_dist = self.calculate_distance(nodes[permutation[j - 1]], depot)
                current_car = current_car + 1
                current_car_capacity = capacity
                cars[current_car] = cars[current_car] + 1
                distance = self.calculate_distance(depot, nodes[permutation[j]])

            overall_distance = overall_distance + distance + wayback_dist

        if wayback_dist == 0:
            wayback_dist = self.calculate_distance(nodes[permutation[j - 1]], depot)
            overall_distance = overall_distance + wayback_dist

        return overall_distance, cars

    def get_neighbors(self, permutation, n):
        neighbors = [0] * int(n)
        for i in range(int(n)):
            neighbors[i] = np.random.permutation(permutation)
        return neighbors

    def existintabu(self, tabu, per):
        tabu = np.array(tabu)
        if per in tabu:
            return 1
        return 0

    def Tabu_Search(self, dimension, capacity, coordesction, demand, depot):

        startt = time.time()

        curr_per = self.generate_per(dimension)
        curr_cost, curr_cars = self.calc_cost(curr_per, dimension, capacity, coordesction, demand, depot)

        n = dimension / 2
        tabulist = [[0]]
        tabucosts = [0]
        tabucars = [[0]]
        tabulist[0] = curr_per
        tabucosts[0] = curr_cost
        tabucars[0] = curr_cars

        iter = 2

        for i in range(iter):

            neighbors = self.get_neighbors(curr_per, n)
            print(neighbors[0])
            print(i)
            best_cost = sys.maxsize
            best_cars = []
            best_per = []
            for j in range(len(neighbors)):
                new_cost, new_cars = self.calc_cost(neighbors[j], dimension, capacity, coordesction, demand, depot)
                if new_cost < best_cost and self.existintabu(tabulist, neighbors[j]) == 0:
                    best_cost = new_cost
                    best_cars = new_cars
                    best_per = neighbors[j]
            tabulist.append(best_per)
            tabucosts.append(best_cost)
            tabucars.append(best_cars)
            curr_per = best_per

        for i in range(len(tabucosts)):
            if tabucosts[i] < best_cost:
                best_per = tabulist[i]
                best_cost = tabucosts[i]
                best_cars = tabucars[i]

        self.print_best_sol(best_per, best_cars)

        elapsed = time.time() - startt
        print("Overall TS runtime: " + str(elapsed))

        return best_per, best_cost


    def choose_next_state_probabilistically(self, dimension, visited):
        #TBD!!
        x = 0
        r = -1
        while x == 0:
            x = 1
            r = random.randrange(0, dimension)
            for i in range(len(visited)):
                if visited[i] == r:
                    x = 0
        return r



    def ACO(self, dimension, capacity, coordsection, demand, depot):

        startt = time.time()

        best_cost = sys.maxsize
        best_cars = []

        x = [0] * Iterations
        fit = [0] * Iterations

        for i in range(Iterations):
            x[i] = i
            visited = []
            start = random.randrange(0, dimension)
            visited.append(start)
            per = [start]

            while len(visited) < dimension:
                next_node = self.choose_next_state_probabilistically(dimension, visited)
                per.append(next_node)
                visited.append(next_node)

            cost, cars = self.calc_cost(per, dimension, capacity, coordsection, demand, depot)

            if cost < best_cost:
                best_per = per
                best_cost = cost
                best_cars = cars

            fit[i] = best_cost

        self.print_best_sol(best_per, best_cars)

        elapsed = time.time() - startt
        print("Overall ACO runtime: " + str(elapsed))

        #plt.title('ACO:')
        #plt.plot(x, fit, label="fitness")
        #plt.legend()
        #plt.show()

        return best_cost


    def CPSO(self, dimension, capacity, coordsection, demand, depot):

        startt = time.time()

        particles = [] * GA_POPSIZE
        BEST = 0
        found = False

        iter = 100

        x = [0] * 100
        fit = [0] * 100


        for i in range(iter):

            per = self.generate_per(dimension)
            velocity = list(np.random.uniform(size=len(per)))
            particles.append(Particle(per, velocity))

        global_best = self.generate_per(dimension)

        for i in range(100):

            x[i] = i
            c, cars = self.calc_cost(global_best, dimension, capacity, coordsection, demand, depot)
            fit[i] = c

            for particle in particles:

                objective = self.calc_cost(particle.position, dimension, capacity, coordsection, demand, depot)
                if objective == BEST:
                    global_best = particle.position
                    found = True
                    break

                else:
                    curr_fit = self.calc_cost(global_best, dimension, capacity, coordsection, demand, depot)
                    if objective < curr_fit:
                        global_best = particle.position

                    curr_fit = self.calc_cost(particle.personal_best, dimension, capacity, coordsection, demand, depot)

                    if objective < curr_fit:
                        particle.personal_best = particle.position

            if found:
                break

            w = (0.6 * (i - Iterations) / (Iterations ** 2)) + 0.6
            c1 = -2 * i / Iterations + 2.5
            c2 = 2 * i / Iterations + 0.5

            # for each particle, update velocity, and update position
            for particle in particles:
                particle.velocity_update(w, c1, c2, global_best)
                particle.position_update()


        global_cost, cars = self.calc_cost(global_best, dimension, capacity, coordsection, demand, depot)

        elapsed = time.time() - startt
        print("Overall CPSO runtime: " + str(elapsed))

        #plt.title('PSO:')
        #plt.plot(x, fit, label="fitness")
        #plt.legend()
        #plt.show()

        return global_best , global_cost








if __name__ == "__main__":
    file = 'E-n22-k4.txt'
    C = ReadData()
    dimension, capacity, nodes, demand, depot = C.read_data(file)
    print(dimension)
    print(capacity)
    print(nodes)
    print(demand)
    problem = CVRP()

    print("Simulated Annealing:")

    cost = problem.Simulated_Annealing(dimension, capacity, nodes, demand, depot)
    print(cost)

    print("Genetic Algorithm:")

    p, cd = problem.Genetic_Algorithm(dimension, capacity, nodes, demand, depot)
    print(p)
    print(cd)

    print("Tabu Search:")

    p3, c3 = problem.Tabu_Search(dimension, capacity, nodes, demand, depot)
    print(p3)
    print(c3)

    print("ACO:")

    p4 = problem.ACO(dimension, capacity, nodes, demand, depot)
    print(p4)

    print("CPSO:")

    p5, c5 = problem.CPSO(dimension, capacity, nodes, demand, depot)
    print(p5)
    print(c5)

