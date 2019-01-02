from my.Bandit import Bandit
from my.EpsilonGreedyAgent import EpsilonGreedyAgent
import numpy as np


class Environment:
    # bandits : The slot machines.
    # agent   : The agent.
    def __init__(self, bandits, agent):
        # The slot machines.
        self.bandits = bandits

        # The agent.
        self.agent = agent

    # Runs the experiment.
    # n : Number of bets.
    def run(self, n):
        outcomes = np.empty(n)
        for i in range(n):
            j = self.agent.choose_action();
            winning_bet = self.bandits[j].play()
            self.agent.update(j, winning_bet)
            outcomes[i] = 1 if winning_bet else 0
        return outcomes
