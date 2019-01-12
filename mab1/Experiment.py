from Bandit import Bandit
from Agent import Agent
from Environment import Environment


class Experiment:
    @staticmethod
    def main():
        reels: int = 3
        symbols: int = 20
        delta: float = 0
        n: int = 100000

        bandit: Bandit = Bandit(reels, symbols, delta)
        agent: Agent = Agent()
        environment: Environment = Environment(bandit, agent)

        for _ in range(n):
            environment.doInteraction()

        est_pwin: float = agent.estimate()
        print('Estimated pwin: %f' % est_pwin)


if __name__ == "__main__":
    Experiment.main()
