import random
import sys
from copy import deepcopy, copy
import numpy
from matplotlib import pyplot as plt
import Dummies
from MegaHal import MegaHal
from RoshamboPlayer import RoshamboPlayer


POPSIZE = 1000
num_of_opponents = 20
ELITRATE = 0.1
Iterations = 100

class Host_struct:

    def __init__(self, host, fitness, total_score):
        self.host = host
        self.fitness = fitness
        self.total_score = total_score


class Agent:

    def __init__(self, matches, hosts_num, num_of_opponents):
        self.hosts_num = hosts_num
        self.num_of_opponents = num_of_opponents
        self.parasites = []
        self.hosts = [Host_struct] * hosts_num
        self.matches = matches

    def initialize_host(self):

        host = [0] * 100

        for i in range(len(host)):
            host[i] = random.randrange(0, 3)

        return host

    def initialize_hosts(self):

        for i in range(POPSIZE):
            curr_host = Host_struct([], 0, 0)
            curr_host.host = self.initialize_host()
            self.hosts[i] = curr_host

        return self.hosts


    def init_opponents(self):

        i = 0
        while i < self.num_of_opponents:

            self.parasites.append(Dummies.Pi())
            self.parasites.append(Dummies.Copy())
            self.parasites.append(Dummies.Flat())
            self.parasites.append(Dummies.Freq())
            self.parasites.append(Dummies.Switch())
            self.parasites.append(Dummies.AntiFlat())
            self.parasites.append(Dummies.Bruijn81())
            self.parasites.append(Dummies.Foxtrot())
            self.parasites.append(Dummies.Play226())
            self.parasites.append(Dummies.RndPlayer())
            self.parasites.append(Dummies.Rotate())
            self.parasites.append(Dummies.SwitchALot())
            i += 13


    def game(self, opponent, host):

        total_score = 0
        # if winner == 1 -> host is the winner, if winner == 0 the opponent is the winner.
        winner = 0

        opponent.newGame(self.matches)

        for i in range(self.matches):
            opponent.storeMove(host.host[i], 0)
            opponent_move = opponent.nextMove()
            host_move = host.host[i]
            if opponent_move != host_move:
                if host_move == (opponent_move + 1) % 3:
                    total_score += 1
                elif opponent_move == (host_move + 1) % 3:
                    total_score -= 1

        if total_score >= self.matches:
            winner = 1

        host.total_score += total_score

        return winner, total_score

    def calc_fitness_for_host(self, host):

        matches_won = 0
        total_wins = 0

        for i in range(num_of_opponents):
            winner, total_score = self.game(self.parasites[i], host)
            # if winner == 1 -> host is the winner, if winner == 0 the opponent is the winner.
            if winner == 1:
                matches_won = matches_won + 1

            total_wins += total_score

        host.fitness = total_wins + matches_won * self.matches

        return host.fitness

    def sort_by_fitness(self, hosts):

        hosts.sort(key=lambda x: - x.fitness)

        return hosts


    def elitism(self, population: list[Host_struct], buffer: list[Host_struct], esize):
        temp = population[:esize].copy()
        buffer[:esize] = temp

        return buffer


    def mutation(self, host):

        i1 = random.randrange(len(host) - 1)
        i2 = random.randrange(len(host) - 1)
        host[i1] = random.randrange(3)
        host[i2] = random.randrange(3)

        return host


    def mate(self, population, buffer):
        esize = int(POPSIZE * ELITRATE)
        buffer = self.elitism(population, buffer, esize)

        # mate the rest
        for i in range(esize, POPSIZE):
            i1 = random.randrange(0, POPSIZE // 2)
            i2 = random.randrange(0, POPSIZE // 2)

            host1 = population[i1].host
            host2 = population[i2].host

            index1 = random.randrange(0, len(host1) - 1)
            index2 = random.randrange(index1, len(host2))

            buffer[i].host = host1[:index1] + host2[index1:index2] + host1[index2:]

            p = random.randrange(0, 100)

            if p < 50:
                buffer[i].host = deepcopy(self.mutation(buffer[i].host))

        return buffer


if __name__ == "__main__":


    
    problem = Agent(matches=100, hosts_num=1000, num_of_opponents=20)

    problem.init_opponents()
    problem.hosts = problem.initialize_hosts()

    x = [0] * Iterations
    y = [0] * Iterations

    best_solution = copy(problem.hosts[0])
    best_solution.fitness = -sys.maxsize - 1

    buffer = problem.hosts.copy()

    for i in range(Iterations):

        x[i] = i


        for l in range(len(problem.hosts)):
            problem.hosts[l].fitness = problem.calc_fitness_for_host(problem.hosts[l])

        problem.hosts = problem.sort_by_fitness(problem.hosts)

        if problem.hosts[0].fitness > best_solution.fitness:
            best_solution = copy(problem.hosts[0])

        y[i] = best_solution.fitness

        print("best solution for now:")
        print(best_solution.host)
        print(best_solution.fitness)

        problem.mate(problem.hosts, buffer)
        problem.hosts = buffer.copy()

    print("best solution overall: ", best_solution.host, best_solution.fitness)

    plt.title('fitness for best solution per iteration')
    plt.plot(x, y, label="fitness")
    plt.legend()
    plt.show()






