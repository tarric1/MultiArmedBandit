from Bandit import Bandit
from Agent import Agent


class Environment:
    def __init__(self, bandit: Bandit, agent: Agent):
        self.bandit = bandit
        self.agent = agent

    def doInteraction(self):
        winning_bet: bool = self.bandit.pull
        self.agent.update(winning_bet)
