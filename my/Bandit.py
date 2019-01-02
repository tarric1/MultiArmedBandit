import numpy as np


class Bandit:
    # reels   : Number of reels of the slot machine.
    # symbols : Number of symbols on each reel.
    # delta   : Difference between the real probability of winning and the theoretical one.
    def __init__(self, reels, symbols, delta):
        # Theoretical probability of winning.
        pwin_th = 1 / np.power(symbols, reels)
        # Real probability of winning.
        self.pwin = pwin_th + delta
        print('pwin_th = {0}, pwin = {1}'.format(pwin_th, self.pwin))

    # Returns True if you win.
    def play(self):
        p = np.random.rand()
        return p < self.pwin
