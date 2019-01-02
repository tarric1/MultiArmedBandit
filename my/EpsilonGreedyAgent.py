from my.Agent import Agent
import numpy as np


class EpsilonGreedyAgent(Agent):
    # bandits_num : Number of slot machines.
    # epsilon     : Epsilon value.
    def __init__(self, bandits_num, epsilon):
        self.bandits_num = bandits_num
        self.epsilon = epsilon
        # Store the number of bets for each slot machine.
        self.bets = np.empty(bandits_num)
        # Store the ration of winning bets for each slot machine.
        self.winning_ratios = np.empty(bandits_num)

    # Chooses on which slot machine to play, by means Epsilon-Greedy algorithm.
    def choose_action(self):
        p = np.random.random()
        if p < self.epsilon:
            # Choose a random slot machine.
            j = np.random.choice(self.bandits_num)
        else:
            # Greedy: Choose the slot machine with max winning ratio.
            j = np.argmax(self.winning_ratios)
        return j

    # Updates the agent data based on the result of the choosen action.
    # i           : Slot machine id.
    # winning_bet : True if the bet has been winning.
    def update(self, j, winning_bet):
        prev = self.bets[j]
        self.bets[j] += 1
        if winning_bet:
            self.winning_ratios[j] = (self.winning_ratios[j] * prev + 1) / self.bets[j]
