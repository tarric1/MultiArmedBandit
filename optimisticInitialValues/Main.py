import matplotlib.pyplot as plt
from optimisticInitialValues.Experiment import Experiment as Exp


real_means = [1, 2, 3, 5]
N = 100000
mean_rewards1 = Exp.run(real_means, 7.5, N)
mean_rewards2 = Exp.run(real_means, 5.5, N)
mean_rewards3 = Exp.run(real_means, 3.5, N)
mean_rewards4 = Exp.run(real_means, 2.5, N)
mean_rewards5 = Exp.run(real_means, 1.5, N)

plt.plot(mean_rewards1, label='upper_limit = 7.5')
plt.plot(mean_rewards2, label='upper_limit = 5.5')
plt.plot(mean_rewards3, label='upper_limit = 3.5')
plt.plot(mean_rewards4, label='upper_limit = 2.5')
plt.plot(mean_rewards5, label='upper_limit = 1.5')
plt.xscale('log')
plt.legend()
plt.show()
