from my.Bandit import Bandit
from my.EpsilonGreedyAgent import EpsilonGreedyAgent
from my.Environment import Environment
import numpy as np
import matplotlib.pyplot as plt

# Slot machines initialisation.
reels = 3
symbols = 10
deltas = [0.0000125, 0.000025, 0.00005, 0.0001]
bandits = [Bandit(reels, symbols, delta) for delta in deltas]

# Agent initialisation...
epsilon = 0.2
agent = EpsilonGreedyAgent(len(bandits), epsilon)

# Environment initialisation.
environment = Environment(bandits, agent)

# The experiment starts.
n = 10000
outcomes = environment.run(n)
winning_bets = np.sum(outcomes);
print('bets = {0}'.format(n));
print('winning_bets = {0}'.format(winning_bets))
print('winning_ratio = {0}'.format(winning_bets / n))
print('max pwin (best winning ratio) = {0}'.format(np.max([bandit.pwin for bandit in bandits])))

# View the experiment results.
rewards = np.cumsum(outcomes) / np.arange(1, n + 1)
plt.plot(rewards, label='epsilon = {0}'.format(epsilon))

# Plot winning probabilities.
for bandit in bandits:
    plt.plot(np.array([bandit.pwin] * n), label='pwin = {0}'.format(bandit.pwin))

plt.xscale('linear')
plt.legend()
plt.show()
