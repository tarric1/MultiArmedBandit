from typing import List
from src.base.Bandit import Bandit
from src.epsilongreedy.Player import Player
import numpy as np
import matplotlib.pyplot as plt


class Experiment:
    @staticmethod
    def main():
        reels: int = 3
        symbols: int = 10
        deltas: List[float] = [0.0002, 0.0001, -0.0001]
        epsilon: float = 0.05
        n: int = 1000000

        bandits: List[Bandit] = [Bandit(reels, symbols, delta) for delta in deltas]
        player: Player = Player(epsilon, bandits)

        r: List[int] = [0] * n
        for i in range(n):
            r[i] = player.play()

        rewards_trend: List[float] = np.cumsum(r) / np.arange(1, n + 1)
        plt.plot(rewards_trend, label='mean reward = {0:.5f}'.format(rewards_trend[-1]))

        for i in range(len(bandits)):
            plt.plot([0, n - 1], [bandits[i].pwin, bandits[i].pwin], label='pwin = {0:.5f}'.format(bandits[i].pwin))
            print('Bandit #{0} : q = {1:.5f}  pwin = {2:.5f}'.format(i, player.q[i], bandits[i].pwin))

        plt.xscale('log')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    Experiment.main()
