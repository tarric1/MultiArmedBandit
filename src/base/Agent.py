from src.base.Bandit import Bandit


class Agent:
    def __init__(self, bandit: Bandit):
        self.bandit: Bandit = bandit
        self.k: int = 0
        self.q: float = 0

    def update(self, r: float):
        self.q += (r - self.q) / (self.k + 1)
        self.k += 1

    def do(self) -> float:
        r: float = self.bandit.interact()
        self.update(r)
        return r
