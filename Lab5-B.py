import random
import sys
from collections import Counter
from copy import deepcopy, copy
from sklearn import preprocessing
import pandas
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

POPSIZE = 8
Iterations = 8
ELITRATE = 0.1

class Data:

    def readdata(self,):

        filename = 'glass.csv'

        dataframe = pandas.read_csv(filename, header=None)

        print(dataframe.shape)

        target = dataframe.values[:, -1]
        counter = Counter(target)

        for k, v in counter.items():
            per = v / len(target) * 100
            print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))

        data = dataframe.values[:, 1:-1]
        #print(dataframe)
        #print(data)

        return data


    def split_data(self, data):
        label = data.values[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(data, label, stratify=label, random_state=1, test_size=0.2)

        print("training:")
        print("X:")
        print(x_train)
        print("Y:")
        print(y_train)

        print("test:")
        print("X:")
        print(x_test)
        print("Y:")
        print(y_test)

        return x_train, x_test, y_train, y_test


    def normalize_data(self, data):
        # normlizing data:
        print("before")
        print(data)
        print("after:")

        scaler = preprocessing.MinMaxScaler()
        d = scaler.fit_transform(data)
        scaled_data = pandas.DataFrame(d)
        print(scaled_data)

        return scaled_data

    def normalize_data2(self, data):


        for i in range(8):
            minimum = 100.0
            maximum = 0.0

            for j in range(len(data)):
                minimum = min(minimum, data[j][i])
                maximum = max(maximum, data[j][i])

            for j in range(len(data)):
                data[i][j] = (data[j][i] - minimum) / (maximum - minimum)

        print(Data)

        return data

    def softmax(self, x):

        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def MLP(self, x_train, x_test, y_train, y_test):

        print("MLP:")

        clf = MLPClassifier(random_state=1, max_iter=3000).fit(x_train, y_train)

        print(clf)
        print(clf.predict_proba(x_test[:1]))
        print(self.softmax(clf.predict_proba(x_test[:1])))
        print(clf.predict(x_test))
        ypred = clf.predict(x_test)
        print(y_test)
        print(clf.score(x_test, y_test))
        print(ypred)

        return ypred



class CLS:


    def retrieve_dataset(self):
        f = open('glass.csv', 'r')

        nn_input = []
        nn_output = []

        for line in f.readlines():
            elements = line.split(',')
            nn_input.append([float(x) for x in elements[1:-1]])
            dict = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5}
            nn_output.append(dict[int(elements[-1][0])])

        return nn_input, nn_output

    def divide_dataset(self, nn_input, nn_output):

        classes = [[], [], [], [], [], []]

        train_input, train_output, test_input, test_output = [], [], [], []

        for i in range(len(nn_input)):
            classes[nn_output[i]].append(nn_input[i])

        for i in range(6):

            test_size = int(0.2 * len(classes[i]))
            for j in range(test_size):
                test_input.append(classes[i][j])
                test_output.append(i)

            for j in range(test_size + 1, len(classes[i])):
                train_input.append(classes[i][j])
                train_output.append(i)

        return train_input, train_output, test_input, test_output

    def normalize_dataset(self, nn_input):

        for j in range(9):
            f_min, f_max = 100.0, 0.00
            f_max = max(nn_input[:][j])

            for i in range(len(nn_input)):
                f_min = min(f_min, nn_input[i][j])
                f_max = max(f_max, nn_input[i][j])

            for i in range(len(nn_input)):
                nn_input[i][j] = (nn_input[i][j] - f_min) / (f_max - f_min)

        return nn_input

    def preprocess_data(self):
        nn_input, nn_output = self.retrieve_dataset()
        nn_input = self.normalize_dataset(nn_input)
        train_input, train_output, test_input, test_output = self.divide_dataset(nn_input, nn_output)
        return train_input, train_output, test_input, test_output





class Network:

    def __init__(self, depth, hidden, activation):
        self.depth = depth
        self.hidden = hidden
        self.activation = activation


class Genetic_struct:

    def __init__(self, network, fitness, reg):
        self.network = network
        self.fitness = fitness
        self.reg = reg


class Algorithm:

    def init_population(self):

        population = []

        for i in range(POPSIZE):
            h = []
            r = random.randrange(1, 10)
            depth = r
            for i in range(depth):
                tmp = random.randrange(2, 200)
                h.append(tmp)

            prob = random.randrange(0, 2)
            if prob == 0:
                activation = 'relu'
            else:
                activation = 'tanh'

            fitness = np.inf
            curr_network = Network(depth, h, activation)

            citizen = Genetic_struct(curr_network, fitness, 0)
            population.append(citizen)

        return population


    def calculate_regression(self, cls, citizen):

        c = 1
        l = 0.5

        weights = cls.coefs_
        length = len(weights)
        wsum = 0

        for i in range(length):
            helper = len(weights[i])

            for j in range(helper):
                print(weights[i][j])
                curr = weights[i][j] ** 2
                wsum = wsum + curr


        depth = citizen.network.depth

        for k in range(depth):
            d = citizen.network.hidden[k]
            c = c * d

        c = c * 54

        x1 = wsum * l
        x2 = len(x_train) * 2

        reg = x1 / x2 / c

        return reg


    def calc_fit_for_one(self, citizen):
        h = citizen.network.hidden
        itr = 1500
        a = citizen.network.activation
        s = 'adam'

        reg = 0

        cls = MLPClassifier(hidden_layer_sizes=h, max_iter=itr, activation=a, solver=s, random_state=1)

        cls.fit(x_train, y_train)
        predictionY = cls.predict(x_test)

        confusion = confusion_matrix(predictionY, y_test)
        overall_sum = confusion.sum()
        d_sum = confusion.trace()

        citizen.fitness = d_sum / overall_sum


        #reg = self.calculate_regression(cls, citizen)
        citizen.reg = reg

        return citizen




    def calculate_fittnes(self, population):

        for i in range(len(population)):

            population[i] = self.calc_fit_for_one(population[i])

        return population


    def sort_by_fitness(self, population):

        population.sort(key=lambda x: - x.fitness)

        return population

    def elitism(self, population: list[Genetic_struct], buffer: list[Genetic_struct], esize):

        temp = population[:esize].copy()
        buffer[:esize] = temp

        return buffer

    def mate(self, population, buffer):

        esize = int(POPSIZE * ELITRATE)
        buffer = self.elitism(population, buffer, esize)

        # mate the rest
        for i in range(esize, POPSIZE):
            i1 = random.randrange(0, POPSIZE - 1)
            i2 = random.randrange(0, POPSIZE - 1)
            print(len(population))
            print(i1)
            print(i2)

            citizen1 = population[i1]
            citizen2 = population[i2]

            index1 = random.randrange(0, citizen1.network.depth)
            index2 = random.randrange(0, citizen2.network.depth)

            buffer[i].network.hidden = citizen1.network.hidden[:index1] + citizen2.network.hidden[index1:index2] + citizen1.network.hidden[index2:]
            buffer[i].network.depth = len(buffer[i].network.hidden)

            p = random.randrange(0, 100)

        return buffer

    def genetic_algorithm(self):

        population = self.init_population()
        buffer = population.copy()

        best_solution = copy(population[0])
        best_fitness = -sys.maxsize - 1

        for i in range(Iterations):
            population = self.calculate_fittnes(population)

            population = self.sort_by_fitness(population)

            if population[0].fitness > best_fitness:
                best_solution = population[0]
                best_fitness = population[0].fitness

            print("best solution for now:")

            print(best_fitness)


            population = self.mate(population, buffer)

        print()


        print("Best solution overall:")
        print("Best solution accuracy:")
        print(best_fitness)

        print('Best solution depth: ', best_solution.network.depth)
        print('Best solution layers: ', best_solution.network.hidden)
        print('Best solution activation: ', best_solution.network.activation)






if __name__ == "__main__":
    problem = Data()
    cls = CLS()

    #data = problem.readdata()
    #problem.split_data(data)
    #normalized_data = problem.normalize_data(data)

    #x_train, x_test, y_train, y_test = problem.split_data(normalized_data)

    #problem.MLP(x_train, x_test, y_train, y_test)

    print("Genetic:")

    x_train, y_train, x_test, y_test = cls.preprocess_data()

    G = Algorithm()
    a = G.genetic_algorithm()



