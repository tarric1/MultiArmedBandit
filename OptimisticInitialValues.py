from Bandit import Bandit
import numpy as np
import matplotlib.pyplot as plt

reels = 3
symbols = 10
bandits = [Bandit(reels, symbols, 5000), Bandit(reels, symbols, 10000), Bandit(reels, symbols, 20000)]
m = len(bandits)

n = 10000
estimated_jackpots = np.empty(m)
estimated_jackpots.fill(30000)
rewards = np.empty(n)
for i in range(n):
    j = np.argmax(estimated_jackpots)

    estimated_jackpots[j] += 1
    reward = bandits[j].play()
    if i == 0:
        rewards[i] = reward - 1
    else:
        rewards[i] = rewards[i - 1] + (reward - 1)
    if reward > 0:
        estimated_jackpots[j] = 0

plt.plot(rewards, label='optimistic-initial-values')
plt.xscale('linear')
plt.legend()
plt.show()
