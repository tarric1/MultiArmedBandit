import numpy as np


class Bandit:
    def __init__(self, reels: int, symbols: int, delta: float):
        self.pwin: float = 1 / np.power(symbols, reels) + delta

    def interact(self) -> float:
        return 1 if np.random.rand() < self.pwin else 0
