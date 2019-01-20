from src.base.Bandit import Bandit


class Player:
    def __init__(self, bandit: Bandit):
        self.bandit: Bandit = bandit
        self.k: int = 0
        self.q: float = 0

    def update(self, r: float):
        self.q += (r - self.q) / (self.k + 1)
        self.k += 1

    def play(self) -> float:
        r: float = self.bandit.pull()
        self.update(r)
        return r
