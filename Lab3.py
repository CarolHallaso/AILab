import math
import random

import numpy as np
from numpy.random import randint
import sys

Iterations = 1000
GA_POPSIZE = 2048
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
            for k in range(num):
                print(permutation[i], end=' ')
                i = i + 1
            j = j + 1
            print()


    def Simulated_Annealing(self, dimension, capacity, coordsection, demand, depot):
        best_sol = [0]
        best_cost = sys.maxsize
        cars_best_sol = []
        cars = [0] * dimension
        current_car = 0
        current_car_capacity = capacity

        for i in range(Iterations):
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

        print(best_sol)
        print(cars_best_sol)
        self.print_best_sol(best_sol, cars_best_sol)

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
        # self.population.sort(key=lambda x: x.fitness-x.age)
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

            #buffer[i].permutation = population[i1].permutation
            new_per = self.GAcrossover(population[i1].permutation, population[i2].permutation)
            buffer[i].permutation = new_per

        return buffer

    def swap(self, population: list[GA_struct], buffer: list[GA_struct]):

        return buffer, population

    def Genetic_Algorithm(self, dimension, capacity, coordsection, demand, depot):

        pop_alpha = [None] * GA_POPSIZE
        pop_beta = [None] * GA_POPSIZE

        pop_alpha = self.init_pop(pop_alpha, dimension)
        buffer = self.init_pop(pop_beta, dimension)

        population = pop_alpha

        for i in range(30):
            population = self.calc_fitness(population, dimension, capacity, coordsection, demand, depot)
            population = self.sort_by_fitness(population)

            buffer = self.mate(population, buffer)  # mate the population
            population, buffer = problem.swap(population, buffer)

        self.print_best_sol(population[0].permutation, population[0].cars)
        return population[0].permutation, population[0].fitness



if __name__ == "__main__":
    file = 'E-n22-k4.txt'
    C = ReadData()
    d, c, n, demand, depot = C.read_data(file)
    print(d)
    print(c)
    print(n)
    print(demand)
    problem = CVRP()

    cost = problem.Simulated_Annealing(d, c, n, demand, depot)
    print(cost)

    p, cd = problem.Genetic_Algorithm(d, c, n, demand, depot)
    print(p)
    print(cd)