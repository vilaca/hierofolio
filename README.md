# Hierofolio

Hierarchical Risk Parity for UCITS ETFs. Build an ETF universe from ISINs,
fetch historical prices into SQLite, and feed the returns into a hierarchical
risk model and portfolio optimizers.

## Setup

Requires Python 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

The tooling is three small scripts around a shared config/DB:

1. **`etf_config.py`** — build the ETF universe YAML from ISINs (via OpenFIGI).
2. **`etf_fetch.py`** — populate the SQLite price DB (ftgo, with a yfinance fallback).
3. **`etf_analyze.py`** — returns and summary statistics from the DB.

```bash
# 1. Add ETFs by ISIN (auto-resolves name, tickers, exchange, FIGI)
./etf_config.py add IE00BM67HK77
./etf_config.py add IE00BM67HK77 IE00BDBRDM35 IE00BKM4GZ66
./etf_config.py list
./etf_config.py update IE00BM67HK77

# 2. Fetch prices into the SQLite DB
./etf_fetch.py                 # all ETFs in the config
./etf_fetch.py IE00BM67HK77    # a single ISIN
./etf_fetch.py --force         # ignore the cache and re-download

# 3. Analyze the stored data
./etf_analyze.py returns              # returns for all ETFs
./etf_analyze.py returns IE00BM67HK77 # returns for a single ISIN
./etf_analyze.py summary              # annualized stats + correlation
```

Defaults: config `etf_universe.yaml`, database `hierofolio.db`, start date
`2018-01-01`. Override with `--config`, `--db`, and `--start` (see each
script's `--help`).

## Risk model

`RiskModel.py` provides `HRPRiskModel` plus the `ConstrainedMVOOptimizer` and
`RobustOptimizer` portfolio optimizers. Feed it the returns from the DB:

```python
from etf_analyze import read_returns
from RiskModel import HRPRiskModel

returns = read_returns("hierofolio.db")

risk_model = HRPRiskModel(
    returns=returns,
    shrinkage_method="constant_correlation",
    shrinkage_intensity=0.3,
    cluster_mode="full",
    linkage_method="ward",
)
```

The optimizers require `cvxpy` (included in `requirements.txt`).

## Tests

```bash
pytest test_riskmodel.py
```

## License

MIT — see [LICENSE](LICENSE).
