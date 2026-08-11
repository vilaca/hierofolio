from typing import Optional, Protocol
import pandas as pd


class SignalModel(Protocol):
    def signal(self, returns: pd.DataFrame) -> tuple[pd.Series, Optional[pd.Series]]:
        """Return (mu, mu_uncertainty). uncertainty=None → optimizer default."""


class HistoricalMeanSignal:
    """Annualized historical mean — exact current behavior."""
    def signal(self, returns: pd.DataFrame) -> tuple[pd.Series, Optional[pd.Series]]:
        mu = (1 + returns.mean()) ** 252 - 1
        return mu, None
