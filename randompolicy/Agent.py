from typing import List
from base.Bandit import Bandit
import numpy as np


class Agent:
    def __init__(self, bandits: List[Bandit]):
        self.bandits: List[Bandit] = bandits
        self.n: int = len(bandits)
        self.k: List[int] = [0] * self.n
        self.q: List[float] = [0] * self.n

    def choose(self) -> int:
        return np.random.choice(self.n)

    def do(self, a: int) -> float:
        return self.bandits[a].interact()

    def update(self, a: int, r: float):
        self.q[a] += (r - self.q[a]) / (self.k[a] + 1)
        self.k[a] += 1

    def play(self) -> int:
        a: int = self.choose()
        r: float = self.do(a)
        self.update(a, r)
        return r
