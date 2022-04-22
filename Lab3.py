import math
import random

import numpy as np
from numpy.random import randint
import sys

Iterations = 1000

class ReadData:

    def read_data(self, file):
        f = open(file, 'r')

        while f.readline().find('TYPE') == -1:
            pass
        line = f.readline()
        # DIMENSION : 22
        (word, points, dimension) = line.split()
        dimension = float(dimension)

        while f.readline().find('EDGE_WEIGHT_TYPE') == -1:
            pass
        line = f.readline()
        (word, points, capacity) = line.split()
        capacity = float(capacity)

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

        return dimension, capacity, nodes_coordinates, demand





if __name__ == "__main__":
    file = 'E-n22-k4.txt'
    C = ReadData()
    d, c, n, demand = C.read_data(file)
    print(d)
    print(c)
    print(n)
    print(demand)
