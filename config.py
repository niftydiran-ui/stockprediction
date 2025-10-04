from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    min_train_days: int = 400        # minimum history before first prediction
    step_days: int = 1               # roll forward by 1 day
    test_days: int = 252             # evaluate last ~1yr by default
    transaction_cost_bp: float = 2.0 # 2 bps per trade (0.02%)
