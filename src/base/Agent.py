from src.base.Bandit import Bandit


class Agent:
    def __init__(self, bandit: Bandit):
        self.bandit: Bandit = bandit
        self.k: int = 0
        self.q: float = 0

    def do(self) -> float:
        return self.bandit.interact()

    def update(self, r: float):
        self.q += (r - self.q) / (self.k + 1)
        self.k += 1

    def play(self) -> int:
        r: float = self.do()
        self.update(r)
        return r
