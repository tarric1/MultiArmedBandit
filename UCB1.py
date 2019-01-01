from Bandit import Bandit
import numpy as np
import matplotlib.pyplot as plt

reels = 3
symbols = 10
bandits = [Bandit(reels, symbols, 5000), Bandit(reels, symbols, 10000), Bandit(reels, symbols, 20000)]
m = len(bandits)

n = 10000
estimated_jackpots = np.empty(m)
bets = np.empty(m)
rewards = np.empty(n)
for i in range(n):
    j = np.argmax([(estimated_jackpots[k] + np.sqrt(2 * np.log(i + 1) / (bets[k] + 0.001))) for k in range(m)])

    estimated_jackpots[j] += 1
    bets[j] += 1
    reward = bandits[j].play()
    if i == 0:
        rewards[i] = reward - 1
    else:
        rewards[i] = rewards[i - 1] + (reward - 1)
    if reward > 0:
        estimated_jackpots[j] = 0
        bets[j] = 0

plt.plot(rewards, label='optimistic-initial-values')
plt.xscale('linear')
plt.legend()
plt.show()
