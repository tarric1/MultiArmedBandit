from typing import List
from src.base.Bandit import Bandit
from src.base.Agent import Agent
import numpy as np
import matplotlib.pyplot as plt


class Experiment:
    @staticmethod
    def main():
        reels: int = 3
        symbols: int = 10
        delta: float = 0.0002
        n: int = 1000000

        bandit: Bandit = Bandit(reels, symbols, delta)
        agent: Agent = Agent(bandit)

        r: List[float] = [0] * n
        for i in range(n):
            r[i] = agent.play()

        rewards_trend: List[float] = np.cumsum(r) / np.arange(1, n + 1)
        plt.plot(rewards_trend, label='mean reward = {0:.5f}'.format(rewards_trend[-1]))

        plt.plot([0, n - 1], [bandit.pwin, bandit.pwin], label='pwin = {0:.5f}'.format(bandit.pwin))
        print('q = {0:.5f}  pwin = {1:.5f}'.format(agent.q, bandit.pwin))

        plt.xscale('linear')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    Experiment.main()
