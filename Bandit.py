import numpy as np


class Bandit:
    def __init__(self, reels, symbols, jackpot):
        self.jackpot = jackpot
        self.pwin = 1 / np.power(symbols, reels)

    def play(self):
        self.jackpot += 1
        p = np.random.rand()
        if p < self.pwin:
            reward = self.jackpot
            self.jackpot = 0
            return reward
        else:
            return 0
