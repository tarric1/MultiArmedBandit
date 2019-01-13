import matplotlib.pyplot as plt
from epsilonGreedy.Experiment import Experiment as EpsGreedyExp
from src.optimisticInitialValues.Experiment import Experiment as OptInitValExp
from src.ucb1.Experiment import Experiment as Ucb1Exp

real_means = [1, 2, 3, 5]
N = 100000
eps_greedy_mean_rewards = EpsGreedyExp.run(real_means, 0.02, N)
opt_init_val_mean_rewards = OptInitValExp.run(real_means, 7.5, N)
ucb1_mean_rewards = Ucb1Exp.run(real_means, N)

plt.plot(eps_greedy_mean_rewards, label='epsilon = 0.20')
plt.plot(opt_init_val_mean_rewards, label='upper_limit = 7.5')
plt.plot(ucb1_mean_rewards, label='ucb1')
plt.xscale('log')
plt.legend()
plt.show()

plt.plot(eps_greedy_mean_rewards, label='epsilon = 0.20')
plt.plot(opt_init_val_mean_rewards, label='upper_limit = 7.5')
plt.plot(ucb1_mean_rewards, label='ucb1')
plt.xscale('linear')
plt.legend()
plt.show()
