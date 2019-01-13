import matplotlib.pyplot as plt
from src.ucb1.Experiment import Experiment as Exp


real_means = [1, 2, 3, 5]
N = 100000
mean_rewards = Exp.run(real_means, N)

plt.plot(mean_rewards)
plt.xscale('log')
plt.show()
