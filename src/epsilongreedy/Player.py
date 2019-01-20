from typing import List
from src.base.Bandit import Bandit
import numpy as np


class Player:
    def __init__(self, epsilon: float, bandits: List[Bandit]):
        self.epsilon: float = epsilon
        self.bandits: List[Bandit] = bandits
        self.n: int = len(bandits)
        self.k: List[int] = [0] * self.n
        self.q: List[float] = [0] * self.n

    def choose(self) -> int:
        a: int = 0
        p: float = np.random.random()
        if p < self.epsilon:
            a = np.random.choice(self.n)
        else:
            a = np.argmax(self.q)
        return a

    def update(self, a: int, r: float):
        self.q[a] += (r - self.q[a]) / (self.k[a] + 1)
        self.k[a] += 1

    def play(self) -> float:
        a: int = self.choose()
        r: float = self.bandits[a].pull()
        self.update(a, r)
        return r
