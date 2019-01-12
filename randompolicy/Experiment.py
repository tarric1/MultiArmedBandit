from typing import List
from base.Bandit import Bandit
from randompolicy.Agent import Agent
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

        r: List[float] = [0] * n
        for i in range(n):
            r[i] = agent.play()

        print('q = {0:.5f}'.format(agent.q))
        print('pwin = {0:.5f}'.format([bandit.pwin for bandit in bandits]))

        rewards_trend: List[float] = np.cumsum(r) / np.arange(1, n + 1)
        plt.plot(rewards_trend, label='simulation')
        for bandit in bandits:
            plt.plot([0, n - 1], [bandit.pwin, bandit.pwin], label='pwin = {0}'.format(bandit.pwin))
            print('Bandit #{0} : q = {1:.5f}  pwin = {2:.5f}'.format())
        plt.xscale('linear')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    Experiment.main()
