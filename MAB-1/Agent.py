class Agent:
    def __init__(self):
        self.bets: int = 0
        self.winning_bets: int = 0

    def update(self, winning_bet: bool):
        self.bets += 1
        if winning_bet:
            self.winning_bets += 1

    def estimate(self) -> float:
        return float(self.winning_bets) / self.bets
