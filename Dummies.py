import random

from RPlayer import RPlayer


class AntiFlat(RPlayer):
    """
    ************************************************************************
    maximally exploit flat distribution
    ************************************************************************
    """

    def __init__(self):
        super().__init__()
        self.moves = []

    def newGame(self, trials):

        self.moves = [0] * 3

    def storeMove(self, move, score):

        self.moves[move % 3] += 1

    def nextMove(self):

        if self.moves[self.ROCK] < self.moves[self.PAPER] and self.moves[self.ROCK] < self.moves[self.SCISSORS]:
            return self.PAPER

        if self.moves[self.PAPER] < self.moves[self.ROCK] and self.moves[self.PAPER] < self.moves[
            self.SCISSORS]:
            return self.SCISSORS

        if self.moves[self.SCISSORS] < self.moves[self.ROCK] and self.moves[self.SCISSORS] < self.moves[
            self.PAPER]:
            return self.ROCK

        if self.moves[self.ROCK] == self.moves[self.PAPER] and self.moves[self.ROCK] < self.moves[
            self.SCISSORS]:
            return self.PAPER

        if self.moves[self.ROCK] == self.moves[self.SCISSORS] and self.moves[self.ROCK] < self.moves[self.PAPER]:
            return self.ROCK

        if self.moves[self.PAPER] == self.moves[self.SCISSORS] and self.moves[self.PAPER] < self.moves[
            self.ROCK]:
            return self.moves[self.SCISSORS]

        return random.randrange(3)

    def getName(self):
        return "Anti Flat Player"

    def getAuthor(self):
        return "standard"


class Copy(RPlayer):
    """
   ************************************************************************
   This player reacts to the opponent's last move
   ************************************************************************
   """

    def __init__(self):
        super().__init__()
        self.lastmove = 0

    def newGame(self, trials):
        self.lastmove = 0

    def storeMove(self, move, score):
        self.lastmove = move

    def nextMove(self):
        return (self.lastmove + 1) % 3

    def getName(self):
        return "Copy Player"

    def getAuthor(self):
        return "standard"


class Freq(RPlayer):
    """
  ************************************************************************
  This player reacts on the most frequently played move of the opponent
  ************************************************************************
  """

    def __init__(self):
        super().__init__()
        self.moves = []

    def newGame(self, trials):
        self.moves = [0] * 3

    def storeMove(self, move, score):
        self.moves[move % 3] += 1

    def nextMove(self):

        if self.moves[self.ROCK] > self.moves[self.PAPER] and self.moves[self.ROCK] > self.moves[self.SCISSORS]:
            return self.PAPER
        elif self.moves[self.PAPER] > self.moves[self.SCISSORS]:
            return self.SCISSORS
        else:
            return self.ROCK

    def getName(self):
        return "Freq Player"

    def getAuthor(self):
        return "standard"


class Flat(RPlayer):
    """
    ************************************************************************
    flat distribution, 20% chance of most frequent actions
    ************************************************************************
    """

    def __init__(self):
        super().__init__()
        self.moves = []
        self.lastmove = 0

    def coin_flip(self, pr_rock, pr_paper):

        p = random.random()
        if p < pr_rock:
            return self.ROCK
        if p < pr_rock + pr_paper:
            return self.PAPER
        return self.SCISSORS

    def newGame(self, trials):

        self.moves = [0] * 3
        self.lastmove = 0

    def storeMove(self, move, score): pass

    def nextMove(self):

        self.moves[self.lastmove % 3] += 1

        if self.moves[self.ROCK] < self.moves[self.PAPER] and self.moves[self.ROCK] < self.moves[self.SCISSORS]:
            self.lastmove = self.coin_flip(0.8, 0.1)

        if self.moves[self.PAPER] < self.moves[self.ROCK] and self.moves[self.PAPER] < self.moves[self.SCISSORS]:
            self.lastmove = self.coin_flip(0.1, 0.8)

        if self.moves[self.SCISSORS] < self.moves[self.ROCK] and self.moves[self.SCISSORS] < self.moves[self.PAPER]:
            self.lastmove = self.coin_flip(0.1, 0.1)

        if self.moves[self.ROCK] == self.moves[self.PAPER] and self.moves[self.ROCK] < self.moves[self.SCISSORS]:
            self.lastmove = self.coin_flip(0.45, 0.45)

        if self.moves[self.ROCK] == self.moves[self.SCISSORS] and self.moves[self.ROCK] < self.moves[self.PAPER]:
            self.lastmove = self.coin_flip(0.45, 0.1)

        if self.moves[self.PAPER] == self.moves[self.SCISSORS] and self.moves[self.PAPER] < self.moves[self.ROCK]:
            self.lastmove = self.coin_flip(0.1, 0.45)

        if self.moves[self.ROCK] == self.moves[self.SCISSORS] and self.moves[self.SCISSORS] == self.moves[self.ROCK]:
            self.lastmove = random.randrange(3)

        return self.lastmove

    def getName(self):
        return "Flat Player"

    def getAuthor(self):
        return "standard"


class Foxtrot(RPlayer):
    """
   ************************************************************************
   set pattern: rand prev+2 rand prev+1 rand prev+0, repeat
   ************************************************************************
   """

    def __init__(self):
        super().__init__()
        self.lastmove = 0
        self.turn = 0

    def newGame(self, trials):
        self.lastmove = 0
        self.turn = 0

    def storeMove(self, move, score): pass

    def nextMove(self):

        self.turn += 1

        if self.turn % 2 == 1:
            self.lastmove = random.randrange(3)
        else:
            self.lastmove = (self.lastmove + self.turn) % 3

        return self.lastmove

    def getName(self):
        return "Foxtrot Player"

    def getAuthor(self):
        return "standard"


class Bruijn81(RPlayer):
    """
   ************************************************************************
   several De Bruijn strings of length 81 concatenated
   ************************************************************************
   """

    def __init__(self):
        super().__init__()
        self.index = 0
        self.db_table = [1, 0, 2, 0, 0, 2, 0, 2, 0, 1, 1, 0, 0, 2, 2, 1, 0, 0, 1, 1, 2, 2, 0, 0, 1, 2, 1, 0, 2, 2, 2, 2, 0, 1, 2, 0, 2,
         2, 0, 2, 1, 1, 2, 1, 1, 0, 1, 1, 1, 2, 0, 0, 0, 0, 2, 1, 0, 1, 0, 1, 2, 2, 1, 2, 0, 1, 0, 0, 0, 1, 0, 2, 1, 2, 1, 2, 2,
         2, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 2, 2, 2, 0, 0, 2, 2, 0, 2, 0, 1, 0, 1, 1, 0, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 0,
         1, 2, 2, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 2, 2, 1, 0, 0, 1, 2, 1, 2, 2, 0, 1, 1, 2, 1, 1, 0, 0, 2, 1, 0, 1, 2, 0, 2,
         1, 2, 1, 0, 2, 1, 1, 2, 0, 0, 1, 0, 1, 2, 2, 0, 1, 0, 0, 2, 0, 1, 2, 0, 1, 1, 2, 1, 1, 1, 1, 0, 2, 0, 2, 1, 0, 2, 2, 0,
         2, 2, 2, 2, 0, 0, 0, 1, 2, 1, 2, 2, 2, 1, 1, 0, 1, 1, 0, 0, 0, 0, 2, 1, 2, 0, 2, 0, 0, 2, 2, 1, 0, 0, 1, 1, 1, 2, 2, 1,
         2, 1, 0, 1, 0, 2, 1, 0, 1, 0, 2, 0, 2, 0, 0, 1, 2, 2, 2, 0, 2, 1, 0, 0, 1, 1, 1, 2, 2, 1, 1, 0, 2, 2, 0, 0, 0, 2, 2, 2,
         2, 1, 2, 2, 0, 1, 2, 0, 0, 2, 0, 1, 1, 2, 1, 2, 1, 1, 1, 1, 0, 0, 2, 1, 2, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 2, 1, 0,
         2, 1, 1, 2, 0, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 2, 0, 2, 2, 2, 1, 2, 1, 0, 2, 1, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 0, 1, 0, 0,
         1, 1, 1, 2, 1, 1, 0, 1, 2, 2, 2, 2, 0, 0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 1, 2, 0, 0, 0, 1, 0, 2, 2, 0, 2, 1, 2,
         2, 0, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 0, 1, 1, 1, 1, 0, 2, 0, 2, 1, 2, 0, 2, 2, 1, 2, 2, 2, 1, 1,
         1, 2, 1, 2, 1, 0, 0, 2, 0, 1, 1, 0, 1, 0, 2, 1, 0, 2, 2, 2, 2, 0, 2, 0, 0, 2, 2, 0, 0, 1, 2, 2, 1, 0, 1, 1, 2, 0, 1, 2,
         1, 1, 2, 2, 0, 1, 0, 1, 2, 2, 2, 0, 2, 0, 0, 2, 0, 2, 1, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 1, 2, 0, 1, 2, 1, 2,
         0, 0, 1, 0, 2, 0, 1, 0, 0, 2, 1, 0, 1, 2, 2, 1, 1, 2, 0, 2, 2, 2, 1, 2, 1, 0, 2, 2, 0, 1, 1, 0, 2, 1, 1, 0, 0, 1, 1, 2,
         1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 0, 0, 0, 0, 1, 1, 0, 2, 1, 2, 1, 2, 2, 2, 0, 0, 1, 2, 0, 1, 0, 1, 2, 1, 1,
         2, 2, 0, 2, 0, 2, 1, 1, 0, 0, 1, 0, 2, 0, 0, 2, 0, 1, 1, 2, 0, 2, 2, 1, 1, 1, 1, 0, 1, 0, 0, 2, 2, 2, 2, 1, 2, 0, 0, 0,
         2, 1, 0, 2, 2, 0, 1, 2, 2, 1, 0, 2, 1, 0, 1, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 2, 1, 2, 2, 2, 2, 1, 2, 0, 0, 0, 1, 1, 2, 0,
         2, 0, 2, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 2, 2, 2, 0, 0, 2, 1, 1, 2, 2, 0, 1, 2, 0, 1, 1, 0, 0, 1, 2, 2, 1, 1, 1, 0, 2,
         0, 1, 0, 2, 2, 0, 2, 2, 1, 0, 2, 1, 2, 2, 2, 1, 0, 1, 0, 2, 2, 1, 2, 0, 2, 1, 0, 2, 0, 0, 0, 0, 1, 2, 1, 0, 0, 2, 0, 2,
         2, 0, 1, 0, 1, 1, 2, 1, 1, 0, 0, 1, 0, 0, 0, 2, 1, 1, 2, 0, 0, 2, 2, 2, 2, 0, 0, 1, 1, 1, 0, 2, 1, 2, 1, 2, 2, 1, 1, 1,
         1, 2, 2, 0, 2, 0, 1, 2, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 0, 1, 2, 1, 2, 2, 1, 0, 0, 2, 0, 2, 1, 0, 1, 0, 2, 2, 0,
         1, 1, 2, 1, 0, 2, 0, 0, 1, 0, 1, 1, 1, 2, 2, 2, 2, 1, 2, 0, 2, 2, 1, 1, 2, 0, 0, 2, 1, 2, 1, 1, 1, 1, 0, 2, 1, 1, 0, 0,
         0, 0, 2, 2, 2, 0, 0, 0, 1, 2, 2, 0, 2, 0, 2, 2, 0, 2, 1, 1, 2, 0, 2, 0, 0, 1, 1, 1, 0, 0, 1, 2, 1, 1, 0, 1, 1, 0, 2, 2,
         0, 0, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 0, 2, 0, 2, 2, 2, 2, 1, 2, 0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 2, 0, 1, 1, 2, 2, 2,
         0, 1, 2, 2, 1, 0, 1, 0, 1, 2, 0, 1, 0, 2, 1, 0, 2, 0, 2, 1, 1, 1, 0, 0, 2, 2, 2, 0, 1, 1, 2, 2, 1, 2, 0, 0, 0, 1, 0, 1,
         2, 1, 0]

    def newGame(self, trials):
        self.index = 0

    def storeMove(self, move, score): pass

    def nextMove(self):

        self.index += 1

        return self.db_table[self.index % 1000]

    def getName(self):
        return "Bruijn 81 Player"

    def getAuthor(self):
        return "standard"


class Pi(RPlayer):
    """
   ************************************************************************
   base each decision on a digit of pi (skipping 0s)
   ************************************************************************
   """

    def __init__(self):
        super().__init__()
        self.index = -1
        # skipping 0s leaves 1088 digits
        self.pi_table = [
            3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4,3,3,8,3,2,7,9,5,0,2,8,8,4,1,9,7,
            1,6,9,3,9,9,3,7,5,1,0,5,8,2,0,9,7,4,9,4,4,5,9,2,3,0,7,8,1,6,4,0,6,2,8,6,2,0,8,9,
            9,8,6,2,8,0,3,4,8,2,5,3,4,2,1,1,7,0,6,7,9,8,2,1,4,8,0,8,6,5,1,3,2,8,2,3,0,6,6,4,
            7,0,9,3,8,4,4,6,0,9,5,5,0,5,8,2,2,3,1,7,2,5,3,5,9,4,0,8,1,2,8,4,8,1,1,1,7,4,5,0,
            2,8,4,1,0,2,7,0,1,9,3,8,5,2,1,1,0,5,5,5,9,6,4,4,6,2,2,9,4,8,9,5,4,9,3,0,3,8,1,9,
            6,4,4,2,8,8,1,0,9,7,5,6,6,5,9,3,3,4,4,6,1,2,8,4,7,5,6,4,8,2,3,3,7,8,6,7,8,3,1,6,
            5,2,7,1,2,0,1,9,0,9,1,4,5,6,4,8,5,6,6,9,2,3,4,6,0,3,4,8,6,1,0,4,5,4,3,2,6,6,4,8,
            2,1,3,3,9,3,6,0,7,2,6,0,2,4,9,1,4,1,2,7,3,7,2,4,5,8,7,0,0,6,6,0,6,3,1,5,5,8,8,1,
            7,4,8,8,1,5,2,0,9,2,0,9,6,2,8,2,9,2,5,4,0,9,1,7,1,5,3,6,4,3,6,7,8,9,2,5,9,0,3,6,
            0,0,1,1,3,3,0,5,3,0,5,4,8,8,2,0,4,6,6,5,2,1,3,8,4,1,4,6,9,5,1,9,4,1,5,1,1,6,0,9,
            4,3,3,0,5,7,2,7,0,3,6,5,7,5,9,5,9,1,9,5,3,0,9,2,1,8,6,1,1,7,3,8,1,9,3,2,6,1,1,7,
            9,3,1,0,5,1,1,8,5,4,8,0,7,4,4,6,2,3,7,9,9,6,2,7,4,9,5,6,7,3,5,1,8,8,5,7,5,2,7,2,
            4,8,9,1,2,2,7,9,3,8,1,8,3,0,1,1,9,4,9,1,2,9,8,3,3,6,7,3,3,6,2,4,4,0,6,5,6,6,4,3,
            0,8,6,0,2,1,3,9,4,9,4,6,3,9,5,2,2,4,7,3,7,1,9,0,7,0,2,1,7,9,8,6,0,9,4,3,7,0,2,7,
            7,0,5,3,9,2,1,7,1,7,6,2,9,3,1,7,6,7,5,2,3,8,4,6,7,4,8,1,8,4,6,7,6,6,9,4,0,5,1,3,
            2,0,0,0,5,6,8,1,2,7,1,4,5,2,6,3,5,6,0,8,2,7,7,8,5,7,7,1,3,4,2,7,5,7,7,8,9,6,0,9,
            1,7,3,6,3,7,1,7,8,7,2,1,4,6,8,4,4,0,9,0,1,2,2,4,9,5,3,4,3,0,1,4,6,5,4,9,5,8,5,3,
            7,1,0,5,0,7,9,2,2,7,9,6,8,9,2,5,8,9,2,3,5,4,2,0,1,9,9,5,6,1,1,2,1,2,9,0,2,1,9,6,
            0,8,6,4,0,3,4,4,1,8,1,5,9,8,1,3,6,2,9,7,7,4,7,7,1,3,0,9,9,6,0,5,1,8,7,0,7,2,1,1,
            3,4,9,9,9,9,9,9,8,3,7,2,9,7,8,0,4,9,9,5,1,0,5,9,7,3,1,7,3,2,8,1,6,0,9,6,3,1,8,5,
            9,5,0,2,4,4,5,9,4,5,5,3,4,6,9,0,8,3,0,2,6,4,2,5,2,2,3,0,8,2,5,3,3,4,4,6,8,5,0,3,
            5,2,6,1,9,3,1,1,8,8,1,7,1,0,1,0,0,0,3,1,3,7,8,3,8,7,5,2,8,8,6,5,8,7,5,3,3,2,0,8,
            3,8,1,4,2,0,6,1,7,1,7,7,6,6,9,1,4,7,3,0,3,5,9,8,2,5,3,4,9,0,4,2,8,7,5,5,4,6,8,7,
            3,1,1,5,9,5,6,2,8,6,3,8,8,2,3,5,3,7,8,7,5,9,3,7,5,1,9,5,7,7,8,1,8,5,7,7,8,0,5,3,
            2,1,7,1,2,2,6,8,0,6,6,1,3,0,0,1,9,2,7,8,7,6,6,1,1,1,9,5,9,0,9,2,1,6,4,2,0,1,9,8,
            9,3,8,0,9,5,2,5,7,2,0,1,0,6,5,4,8,5,8,6,3,2,7,8,8,6,5,9,3,6,1,5,3,3,8,1,8,2,7,9,
            6,8,2,3,0,3,0,1,9,5,2,0,3,5,3,0,1,8,5,2,9,6,8,9,9,5,7,7,3,6,2,2,5,9,9,4,1,3,8,9,
            1,2,4,9,7,2,1,7,7,5,2,8,3,4,7,9,1,3,1,5,1,5,5,7,4,8,5,7,2,4,2,4,5,4,1,5,0,6,9,5,
            9,5,0,8,2,9,5,3,3,1,1,6,8,6,1,7,2,7,8,5,5,8,8,9,0,7,5,0,9,8,3,8,1,7,5,4,6,3,7,4,
            6,4,9,3,9,3,1,9,2,5,5,0,6,0,4,0,0,9,2,7,7,0,1,6,7,1,1,3,9,0,0,9,8,4,8,8,2,4,0,1]

    def newGame(self, trials):
        self.index = -1

    def storeMove(self, move, score): pass

    def nextMove(self):

        if self.index < 0:
            self.index = 0
        else:
            self.index = (self.index + 1) % 1200
            while self.pi_table[self.index] == 0:
                self.index += 1

        return self.pi_table[self.index] % 3

    def getName(self):
        return "Pi Player"

    def getAuthor(self):
        return "standard"


class Play226(RPlayer):
    """
   ************************************************************************
   fixed strategy (20,20,60): plays rock with p 0.2 paper with p 0.2 ad scissors with p 0.6
   ************************************************************************
   """

    def __init__(self):
        super().__init__()
        self.lastmove = 0

    def coin_flip(self, pr_rock, pr_paper):

        p = random.random()
        if p < pr_rock:
            return self.ROCK
        if p < pr_rock + pr_paper:
            return self.PAPER
        return self.SCISSORS

    def newGame(self, trials): pass

    def storeMove(self, move, score): pass

    def nextMove(self):
        return self.coin_flip(0.2, 0.2)

    def getName(self):
        return "226 Player"

    def getAuthor(self):
        return "standard"


class RndPlayer(RPlayer):
    """
   ************************************************************************
   This player plays randomly with equal probabilities
   ************************************************************************
   """

    def __init__(self):
        super().__init__()

    def newGame(self, trials): pass

    def storeMove(self, move, score): pass

    def nextMove(self):
        return random.randrange(3)

    def getName(self):
        return "Random Player"

    def getAuthor(self):
        return "standard"


class Rotate(RPlayer):
    """
   ************************************************************************
   This player rotates Rock Paper Scissors
   ************************************************************************
   """

    def __init__(self):
        super().__init__()
        self.move = 0

    def newGame(self, trials):
        self.move = 0

    def storeMove(self, move, score): pass

    def nextMove(self):

        self.move += 1
        return self.move % 3

    def getName(self):
        return "Rotating Player"

    def getAuthor(self):
        return "standard"


class Switch(RPlayer):
    """
   ************************************************************************
   This player never repeats the previous pick
   ************************************************************************
   """

    def __init__(self):
        super().__init__()
        self.lastmove = 0

    def coin_flip(self, pr_rock, pr_paper):

        p = random.random()
        if p < pr_rock:
            return self.ROCK
        if p < pr_rock + pr_paper:
            return self.PAPER
        return self.SCISSORS

    def newGame(self, trials):
        self.lastmove = 0

    def storeMove(self, move, score): pass

    def nextMove(self):

        if self.lastmove == self.ROCK:
            self.lastmove = self.coin_flip(0.0, 0.5)
        elif self.lastmove == self.PAPER:
            self.lastmove = self.coin_flip(0.5, 0)
        else:
            self.lastmove = self.coin_flip(0.5, 0.5)

        return self.lastmove

    def getName(self):
        return "Switching Player"

    def getAuthor(self):
        return "standard"


class SwitchALot(RPlayer):
    """
   ************************************************************************
   seldom repeat the previous pick
   ************************************************************************
   """

    def __init__(self):
        super().__init__()
        self.lastmove = 0

    def coin_flip(self, pr_rock, pr_paper):

        p = random.random()
        if p < pr_rock:
            return self.ROCK
        if p < pr_rock + pr_paper:
            return self.PAPER
        return self.SCISSORS

    def newGame(self, trials):
        self.lastmove = 0

    def storeMove(self, move, score):
        pass

    def nextMove(self):

        if self.lastmove == self.ROCK:
            self.lastmove = self.coin_flip(0.12, 0.44)
        elif self.lastmove == self.PAPER:
            self.lastmove = self.coin_flip(0.44, 0.12)
        else:
            self.lastmove = self.coin_flip(0.44, 0.44)

        return self.lastmove

    def getName(self):
        return "Switch a Lot Player"

    def getAuthor(self):
        return "standard"