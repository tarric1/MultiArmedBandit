import matplotlib.pyplot as plt
from epsilonGreedy.Experiment import Experiment as EpsGreedyExp
from optimisticInitialValue.Experiment import Experiment as OptInitValExp

real_means = [1, 2, 3, 5]
N = 100000
eps_greedy_mean_rewards = EpsGreedyExp.run(real_means, 0.02, N)
opt_init_val_mean_rewards = OptInitValExp.run(real_means, 7.5, N)

plt.plot(eps_greedy_mean_rewards, label='epsilon = 0.20')
plt.plot(opt_init_val_mean_rewards, label='upper_limit = 7.5')
plt.xscale('log')
plt.legend()
plt.show()
