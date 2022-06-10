from abc import ABC, abstractmethod


class RPlayer(ABC):

    def __init__(self):
        self.ROCK = 0
        self.PAPER = 1
        self.SCISSORS = 2
        self.DRAW = 0
        self.WIN = 1
        self.LOSS = -1

    @abstractmethod
    def newGame(self, trials): pass

    # Store the opponent's choice
    # move is one of ROCK, PAPER, SCISSORS. score is one of DRAW, WIN, LOSS.
    @abstractmethod
    def storeMove(self, move, score): pass

    # Produce your next move. result: one of ROCK, PAPER, SCISSORS.
    @abstractmethod
    def nextMove(self): pass

    # Produce the name (and version) of this player.
    @abstractmethod
    def getName(self): pass


    @abstractmethod
    def getAuthor(self): pass

