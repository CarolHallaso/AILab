import random
import time
from RPlayer import RPlayer


def halbot_compare(context1, context2):
    # Returns An integer which is positive if value1<value2 ...

    prob1, prob2 = 0.0, 0.0

    if context1.seen > 0:
        prob1 = (context1.best - context1.worst) / context1.seen

    if context2.seen > 0:
        prob2 = (context2.best - context2.worst) / context2.seen

    if prob1 < prob2: return -1

    if prob1 > prob2: return 1

    if context1.seen == 0 and context2.seen == 0:

        if context1.size < context2.size: return 1

        if context1.size > context2.size: return -1

        return 0

    if context1.size < context2.size: return -1

    if context1.size > context2.size: return 1

    return 0


class MegaHal(RPlayer):

    def __init__(self):
        super().__init__()
        self.ERROR = 0  # emit a warning message if the algorithm averages more than one millisecond per move
        self.BELIEVE = 1  # gives the number of times a context must be observed before being used for prediction
        self.HISTORY = 40  # gives the maximum context size to observe
        self.WINDOW = 300  # gives the size of the sliding window, 0 being infinite.
        self.move = -1
        self.last_move = -1
        self.random_move = -1
        self.trie = []
        self.trie_size = 0
        self.context_size = 0
        self.context = []
        self.sorted = []
        self.memory = [[0] * (self.HISTORY + 2)] * self.WINDOW
        self.remember = 0
        self.start = time.time()
        self.end = time.time()
        self.think = 0.0
        self.node = None
        self.expected = [0] * 3
        # This is the only external information we have about our opponent; it's a history of the game so far.
        global my_history
        my_history = []
        global opp_history
        opp_history = []

    class NODE:
        # Each node of the trie contains frequency information about the moves made at the context represented
        # by the node, and where the sequent nodes are in the array.
        def __init__(self, total=0, move=[0, 0, 0], next=[0, 0, 0]):
            self.total = total
            self.move = move
            self.next = next

    class CONTEXT:
        # The context array contains information about contexts of various lengths, and this is used to select
        # a context to make the prediction.
        def __init__(self, worst=0, best=0, seen=0, size=0, node=0):
            self.worst = worst
            self.best = best
            self.seen = seen
            self.size = size
            self.node = node

    def halbot(self):

        self.start = time.time()
        global my_history
        global opp_history

        # If we've already started playing, evaluate how well we went on our last turn, and update our list which
        # decides which contexts give the best predictions.
        if len(my_history) != 0:

            # We begin by forgetting which contexts performed well in the distant past.

            if self.WINDOW > 0:
                for i in range(self.context_size):

                    if self.WINDOW > i:
                        if self.memory[(self.remember + i - 1) % self.WINDOW][i] >= 0:

                            if self.memory[(self.remember + i - 1) % self.WINDOW][i] == (
                                    (opp_history[my_history[0] - self.WINDOW + i - 1] + 1) % 3):
                                self.context[i].best -= 1

                            if self.memory[(self.remember + i - 1) % self.WINDOW][i] == (
                                    (opp_history[my_history[0] - self.WINDOW + i - 1] + 2) % 3):
                                self.context[i].worst -= 1

                            self.context[i].seen -= 1

            # Clear our dimmest memory.

            if self.WINDOW > 0:
                for i in range(self.context_size):
                    self.memory[self.remember][i] = -1

            # We then remember the contexts which performed the best most recently.

            for i in range(self.context_size):

                if self.context[i].node >= self.trie_size: continue

                if self.context[i].node == -1: continue

                if self.trie[self.context[i].node].total >= self.BELIEVE:

                    for j in range(3):
                        self.expected[j] = self.trie[self.context[i].node].move[(j + 2) % 3] -\
                                           self.trie[self.context[i].node].move[(j + 1) % 3]

                    self.last_move = 0

                    for j in range(1, 3):
                        if self.expected[j] > self.expected[self.last_move]:
                            self.last_move = j

                    if self.last_move == (opp_history[my_history[0]] + 1) % 3:
                        self.context[i].best += 1

                    if self.last_move == (opp_history[my_history[0]] + 2) % 3:
                        self.context[i].worst += 1

                    self.context[i].seen += 1

                    if self.WINDOW > 0:
                        self.memory[self.remember][i] = self.last_move

            if self.WINDOW > 0:
                self.remember = (self.remember + 1) % self.WINDOW

        # Clear the context to the node which always predicts at random, and the root node.

        self.context_size = 2

        # We begin by forgetting the statistics we've previously gathered about the symbol which is now
        # leaving the sliding window.

        if len(my_history) == 0: my_history = [0]

        if 0 < self.WINDOW < my_history[0]:

            m = min(my_history[0] - self.WINDOW + self.HISTORY, my_history[0])

            for i in range(my_history[0] - self.WINDOW, m):

                # Find the node which corresponds to the context.

                self.node = 0

                for j in range(max(my_history[0] - self.WINDOW, 1), i):
                    self.node = self.trie[self.node].next[opp_history[j]]
                    self.node = self.trie[self.node].next[my_history[j]]

                # Update the statistics of this node with the opponents move.
                self.trie[self.node].total -= 1
                self.trie[self.node].move[opp_history[i]] -= 1

        # Build up a context array pointing at all the nodes in the trie which predict what the opponent is going
        # to do for the current history.While doing this, update the statistics of the trie with the last move made
        # by ourselves and our opponent.

        if self.WINDOW > 0:
            x = max(my_history[0] - min(self.WINDOW, self.HISTORY), 0)
        else:
            x = max(my_history[0] - self.HISTORY, 0)

        for i in range(my_history[0], x, -1):

            self.node = 0

            for j in range(i, my_history[0]):
                self.node = self.trie[self.node].next[opp_history[j]]
                self.node = self.trie[self.node].next[my_history[j]]

            # Update the statistics of this node with the opponents move.

            self.trie[self.node].total += 1
            self.trie[self.node].move[opp_history[my_history[0]]] += 1

            # *	Move to this node, making sure that we've allocated it first.

            if self.trie[self.node].next[opp_history[my_history[0]]] == -1:
                self.trie[self.node].next[opp_history[my_history[0]]] = self.trie_size
                self.trie_size += 1
                self.trie.append(self.NODE())
                self.trie[self.trie_size - 1] = self.NODE(0, [0, 0, 0], [-1, -1, -1])

            self.node = self.trie[self.node].next[opp_history[my_history[0]]]

            # Move to this node, making sure that we've allocated it first.

            if self.trie[self.node].next[my_history[my_history[0]]] == -1:
                self.trie[self.node].next[my_history[my_history[0]]] = self.trie_size
                self.trie_size += 1
                # self.trie=(NODE * )realloc(trie, sizeof(NODE) * trie_size)
                self.trie.append(self.NODE())
                self.trie[self.trie_size - 1] = self.NODE(0, [0, 0, 0], [-1, -1, -1])

            self.node = self.trie[self.node].next[my_history[my_history[0]]]

            # Add this node to the context array.
            self.context_size += 1
            self.context[self.context_size - 1].node = self.node
            self.context[self.context_size - 1].size = self.context_size - 2

        # Sort the context array, based upon what contexts have proved successful in the past
        # We sort the context array y looking at the context lengths which most often give the best predictions.
        # If two contexts give the same amount of best predictions, choose the longest one.If two contexts have
        # consistently failed in the past, choose the shortest one.

        for i in range(self.context_size):
            self.sorted[i] = self.context[i]

        from functools import cmp_to_key

        self.sorted.sort(key=cmp_to_key(halbot_compare))

        # Starting at the most likely context, gradually fall-back until we find a context which has been observed at
        # least BELIEVE times.Use this context to predict a probability distribution over the opponents

        self.move = -1

        for i in range(self.context_size):

            if self.sorted[i].node >= self.trie_size: continue
            if self.sorted[i].node == -1: break

            if self.trie[self.sorted[i].node].total >= self.BELIEVE:

                for j in range(3):
                    self.expected[j] = self.trie[self.sorted[i].node].move[(j + 2) % 3] - \
                                       self.trie[self.sorted[i].node].move[(j + 1) % 3]

                self.move = 0

                for j in range(1, 3):
                    if self.expected[j] > self.expected[self.move]:
                        self.move = j
                break

        # If no prediction was possible, make a random move.

        random_move = random.randrange(3)

        if self.move == -1:
            self.move = random_move

        # Update the timer, and warn if we've exceeded one second per one thousand turns.

        self.end = time.time()

        if self.end - self.start > 1000:
            print('MegaHAL-Infinite is too slow! ***\n\n')

        return self.move

    def newGame(self, trials):

        global my_history
        global opp_history

        if len(self.trie) == 0:

            # If this is the first game we've played, initialise the memory.

            self.context = [self.CONTEXT()] * (self.HISTORY + 2)

            self.sorted = [self.CONTEXT()] * (self.HISTORY + 2)

            # Clear the memory matrix.
            if self.WINDOW > 0:
                self.memory = [[-1] * (self.HISTORY + 2)] * self.WINDOW

        # Clear the trie, by setting its size to unity, and clearing the statistics of the root node.

        self.trie_size = 1
        self.trie = [self.NODE(0, [0, 0, 0], [-1, -1, -1])]

        # Clear the context array.

        for i in range(self.HISTORY + 2):
            self.context[i] = self.CONTEXT(0, 0, 0, 0, 0)

        self.context[0] = self.CONTEXT(0, 0, 0, -1, -1)

        # Clear the variable we use to keep track of how long MegaHAL spends thinking.

        self.think = 0

    def storeMove(self, move, score):
        global opp_history
        opp_history.append(move)
        pass

    def nextMove(self):
        move = self.halbot()
        global my_history
        my_history.append(move)
        return move

    def getName(self):
        return "MegaHal Player"

    def getAuthor(self):
        return "standard"

'''
ttt = MegaHal()
ttt.newGame(4)
c = 0
my_coev = [0, 0, 0, 1, 2, 1, 2, 1, 1, 2, 0, 1, 1, 2, 1, 2, 0, 0, 0, 1, 2, 2, 2, 2, 0, 2, 0, 1, 0, 2, 0, 0, 0, 1, 0, 1, 0, 1, 0, 2, 0, 1, 1, 0, 1, 1, 2, 0, 2, 0, 2, 2, 0, 0, 1, 0, 2, 2, 0, 0, 0, 2, 1, 2, 1, 2, 0, 1, 1, 2, 1, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 2, 0, 2, 2, 2, 0, 0, 1, 0, 2, 0, 2, 2, 1, 0, 2, 2]
paper = 1

for i in range(100):

    my_move = 1                 #set move to paper
    ttt.storeMove(my_move, 1)
    opp_move = ttt.nextMove()

    if opp_move == (my_move + 1)%3: # if AI wins round
        c +=1

print('rounds won:', c)
'''
