import matplotlib.pyplot as plt
from epsilonGreedy.Experiment import Experiment as Exp

real_means = [1, 2, 3, 5]
N = 100000
mean_rewards1 = Exp.run(real_means, 0.20, N)
mean_rewards2 = Exp.run(real_means, 0.10, N)
mean_rewards3 = Exp.run(real_means, 0.05, N)
mean_rewards4 = Exp.run(real_means, 0.02, N)
mean_rewards5 = Exp.run(real_means, 0.01, N)

plt.plot(mean_rewards1, label='epsilon = 0.20')
plt.plot(mean_rewards2, label='epsilon = 0.10')
plt.plot(mean_rewards3, label='epsilon = 0.05')
plt.plot(mean_rewards4, label='epsilon = 0.02')
plt.plot(mean_rewards5, label='epsilon = 0.01')
plt.xscale('log')
plt.legend()
plt.show()
