from typing import List
from mab2.Bandit import Bandit
import numpy as np


class Agent:
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

    def do(self, a: int) -> int:
        return self.bandits[a].interact()

    def update(self, a: int, r: int):
        self.q[a] += (r - self.q[a]) / (self.k[a] + 1)
        self.k[a] += 1

    def play(self) -> int:
        a: int = self.choose()
        r: int = self.do(a)
        self.update(a, r)
        return r
