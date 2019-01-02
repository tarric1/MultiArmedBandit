from abc import ABC, abstractmethod


class Agent(ABC):
    # Chooses on which slot machine to play, by means Epsilon-Greedy algorithm.
    @abstractmethod
    def choose_action(self):
        pass

    # Updates the agent data based on the result of the choosen action.
    # i           : Slot machine id.
    # winning_bet : True if the bet has been winning.
    @abstractmethod
    def update(self, j, winning_bet):
        pass
