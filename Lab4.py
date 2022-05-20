
import random

import numpy as np
from matplotlib import pyplot as plt

Iterations = 100
POPSIZE = 20
POPSIZE2 = 720
ELITRATE = 0.1
host_len = 30

class Host_struct:

    def __init__(self, host, fitness):
        self.host = host
        self.fitness = fitness


class HillisExperiment:

    def generate_host(self, length, r):
        result = [0] * length
        for i in range(length):
            result[i] = random.randrange(0, r)

        return result


    def init_hosts(self, hosts, k):
        for i in range(len(hosts)):
            host = Host_struct([], 0)
            if k == 6:
                hostlen = random.randrange(24, 36)
            else:
                hostlen = random.randrange(120, 300)
            host.host = self.generate_host(hostlen, k)

            hosts[i] = host

        return hosts

    def init_parasites(self, parasites, k):
        for i in range(len(parasites)):
            parasites[i] = random.sample(range(0, 10), k)
        return parasites

    def compare(self, num1, num2, indx1, indx2):
        if indx1 > indx2:
            return num1 < num2
        return num1 > num2

    def swap(self, parasite, indx1, indx2):
        tmp = parasite[indx1]
        parasite[indx1] = parasite[indx2]
        parasite[indx2] = tmp
        return parasite


    def sort(self, parasite, sorting_network):
        i = 0

        while i < (len(sorting_network) - 1):
            indx1 = sorting_network[i]
            indx2 = sorting_network[i+1]
            if self.compare(parasite[indx1], parasite[indx2], indx1, indx2):
                parasite = self.swap(parasite, indx1, indx2)
            i = i + 2

        return parasite

    def check_if_sorted(self, parasite):
        # returns 1 if the parasite is sorted and 0 if not
        for i in range(len(parasite)-1):
            if parasite[i] > parasite[i + 1]:
                return 0
        return 1



    def calc_fitness_for_host(self, host, parasites):
        sorted = 0

        for i in range(len(parasites)):
            p = parasites[i].copy()
            new_parasite = self.sort(p, host.host)
            check = self.check_if_sorted(new_parasite)
            if check == 1:
                sorted = sorted + 1
                #print(new_parasite)
                #print(host.host)

        fitness = sorted / len(parasites)
        host.fitness = -fitness

        return host, fitness

    def calc_fitness_for_parasite(self, parasite, hosts):
        not_sorted = 0

        for i in range(len(hosts)):
            new_parasite = self.sort(parasite, hosts[i].host)
            check = self.check_if_sorted(new_parasite)
            if check == 0:
                not_sorted = not_sorted + 1

        fitness = not_sorted / len(hosts)

        return fitness

    def indirect_replacement_mutation(self, host):

        x = 0
        comp = 0
        while x == 0:
            comp = random.randrange(0, len(host) - 1)
            if comp % 2 == 0:
                x = 1
                #print(comp)

        num1 = host[comp]
        num2 = host[comp + 1]

        j = comp
        while j < len(host) - 2:
            host[j] = host[j + 2]
            j = j + 1

        x = 0
        comp2 = 0
        while x == 0:
            comp2 = random.randrange(0, len(host) - 1)
            if comp2 % 2 == 0:
                x = 1
                #print(comp2)

        j = comp2 + 2
        while j < len(host):
            host[j] = host[j - 2]
            j = j + 1

        host[comp2] = num1
        host[comp2 + 1] = num2

        return host


    def sort_by_fitness(self, hosts):
        # sort the population according to the fitness in ascending order
        hosts.sort(key=lambda x: x.fitness)

        return hosts

    def elitism(self, population, buffer, esize):
        temp = population[:esize].copy()
        buffer[:esize] = temp
        return buffer

    def crossover(self, p1, p2, r):
        #if p1.fitness > p2.fitness:
            #return p1.host, p1.fitness
        #return p2.host, p2.fitness
        result = [0] * len(p1.host)
        for i in range(len(p1.host)):
            result[i] = random.randrange(0, r)
        return result

    def mate(self, population, buffer, k):

        esize = int(POPSIZE2 * ELITRATE)
        buffer = self.elitism(population, buffer, esize)

        # mate the rest
        for i in range(esize, POPSIZE2):

            i1 = random.randrange(0, POPSIZE2 // 2)
            i2 = random.randrange(0, POPSIZE2 // 2)

            new_host = self.crossover(population[i1], population[i2], k)
            buffer[i].host = new_host

        return buffer

    def mutate(self, prob, hosts):
        newhost = Host_struct([], 0)

        for i in range(len(hosts)):
            x = random.random()
            if x < prob:
                newhost.host = problem.indirect_replacement_mutation(hosts[i].host)
                newhost, newfit = problem.calc_fitness_for_host(newhost, parasites)
                if newfit > -hosts[i].fitness:
                    hosts[i].host = newhost.host
                    hosts[i].fitness = newfit

        return hosts

    def mutate2(self, prob, hosts, iteravg):
        newhost = Host_struct([], 0)

        for i in range(len(hosts)):
            x = random.random()
            if hosts[i].fitness < iteravg:
                newhost.host = problem.indirect_replacement_mutation(hosts[i].host)
                newhost, newfit = problem.calc_fitness_for_host(newhost, parasites)
                if newfit > -hosts[i].fitness:
                    hosts[i].host = newhost.host
                    hosts[i].fitness = newfit

            #if x < prob:
            #    newhost.host = problem.indirect_replacement_mutation(hosts[i].host)
             #   newhost, newfit = problem.calc_fitness_for_host(newhost, parasites)
             #   if newfit > -hosts[i].fitness:
             #       hosts[i].host = newhost.host
             #      hosts[i].fitness = newfit

        return hosts


    def print_solution(self, host):

        plt.title("best sorting network:")

        point1 = [0, 0]
        point2 = [0, 0]

        x_values = []
        y_values = []

        for i in range(int(len(host) / 2)):
            point1 = [i + 1, host[i]]
            point2 = [i + 1, host[i + 1]]

            x_values.append(point1[0])
            x_values.append(point2[0])
            y_values.append(point1[1])
            y_values.append(point2[1])

        plt.plot(x_values, y_values)

        #plt.legend()
        plt.show()






if __name__ == "__main__":

    problem = HillisExperiment()

    k = 6
    parasites = [0] * POPSIZE
    hosts = [Host_struct] * POPSIZE2
    buffer = [Host_struct] * POPSIZE2
    parasites = problem.init_parasites(parasites, k)
    print("Parasites:")
    print(parasites)
    hosts = problem.init_hosts(hosts, k)

    print("Hosts:")

    for i in range(len(hosts)):
        print(hosts[i].host)
        print(hosts[i].fitness)

    buffer = hosts.copy()

    #for i in range(len(parasites)):
     #   p = problem.sort(parasites[i], hosts[i])
     #   print(p)

    prob = 0.8

    x = [0] * Iterations
    A = [0] * Iterations
    B = [0] * Iterations

    best_fit = 0
    best_host = []

    for t in range(Iterations):
        print("current iteration:")
        f = 0

        x[t] = t

        for i in range(len(hosts)):
            # calculate the fitness for each host
            tmp_parasites = parasites.copy()
            #print(tmp_parasites)
            host, fit = problem.calc_fitness_for_host(hosts[i], tmp_parasites)
            #print("current host")
            #print(host.host)
            #print(fit)

            f = f + fit

            if fit >= best_fit:
                best_fit = fit
                best_host = host.host

        # calculate the average fitness of this iteration

        avgf = f / len(hosts)
        print("Avg:")
        print(avgf)

        A[t] = -hosts[0].fitness
        B[t] = best_fit

        hosts = problem.sort_by_fitness(hosts)

        print("after sorting:")

        for i in range(len(hosts)):
            print(hosts[i].host)
            print(hosts[i].fitness)

        print("best one in this iteration")

        print(hosts[0].host)
        print(hosts[0].fitness)
        newhost = Host_struct([], 0)

        buffer = problem.mate(hosts, buffer, k)

        hosts = buffer

        hosts = problem.mutate(prob, hosts)

        #prob = prob - 0.5 * prob

        print(hosts[0].host)
        print(hosts[0].fitness)

        #print("last hosts:")
        #for i in range(len(hosts)):
          #  print(hosts[i].host)
          #  print(hosts[i].fitness)

    plt.title('best fittnes per iteration')
    plt.plot(x, B, label="best fitness")
    plt.legend()
    plt.show()


    plt.title('')
    plt.plot(x, A, label="best fitness")
    plt.legend()
    #plt.show()

    print()
    print(best_host)
    print(best_fit)
    print(int(len(best_host) / 2))



    #problem.print_solution(best_host)
