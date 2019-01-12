from typing import List
from mab2.Bandit import Bandit
from mab2.Agent import Agent
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
        agent: Agent = Agent(epsilon, bandits)

        r: List[int] = [0] * n
        for i in range(n):
            r[i] = agent.play()

        rewards_trend: List[float] = np.cumsum(r) / np.arange(1, n + 1)
        plt.plot(rewards_trend, label='simulation')
        for bandit in bandits:
            plt.plot([0, n - 1], [bandit.pwin, bandit.pwin], label='pwin = {0}'.format(bandit.pwin))
        plt.xscale('linear')
        plt.legend()
        plt.show()

        print(agent.q)


if __name__ == "__main__":
    Experiment.main()
