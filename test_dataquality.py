"""Data-quality checks for the price DB.

Unit tests exercise the `quality_report` checker against synthetic good/bad
frames; the integration test runs it on the real hierofolio.db when present
(skipped otherwise) to assert the DB's structural invariants hold.
"""

import os

import numpy as np
import pandas as pd
import pytest

from etf_analyze import quality_report, read_long

DB_PATH = "hierofolio.db"
MAX_GAP_DAYS = 5          # a long weekend + a holiday or two
MAX_ABS_RETURN = 0.5      # no sane broad ETF moves 50% in a day


def clean_frame(days: int = 300, seed: int = 0) -> pd.DataFrame:
    """Two ISINs of positive prices on consecutive business days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=days)
    frames = []
    for isin in ("AAA000000001", "BBB000000002"):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, days)))
        frames.append(pd.DataFrame({"isin": isin, "date": dates, "close": px}))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Unit tests: a clean frame passes, each injected defect is detected
# --------------------------------------------------------------------------

def test_clean_frame_has_no_issues():
    rep = quality_report(clean_frame())
    assert rep['duplicates'] == 0
    assert rep['nulls'] == 0
    assert rep['non_positive'] == 0
    assert rep['weekend_rows'] == 0
    assert rep['max_gap_days'] <= MAX_GAP_DAYS        # Fri->Mon is 3 days
    assert rep['max_abs_return'] < MAX_ABS_RETURN
    assert rep['rows'] == 600


def test_detects_duplicate_keys():
    df = clean_frame()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # repeat one (isin,date)
    assert quality_report(df)['duplicates'] == 1


def test_detects_null_close():
    df = clean_frame()
    df.loc[10, 'close'] = np.nan
    assert quality_report(df)['nulls'] == 1


def test_detects_non_positive_close():
    df = clean_frame()
    df.loc[10, 'close'] = 0.0
    df.loc[11, 'close'] = -5.0
    assert quality_report(df)['non_positive'] == 2


def test_detects_weekend_row():
    df = clean_frame()
    df.loc[len(df)] = {"isin": "AAA000000001", "date": pd.Timestamp("2018-01-06"), "close": 100.0}
    assert quality_report(df)['weekend_rows'] == 1     # 2018-01-06 is a Saturday


def test_detects_large_gap():
    df = clean_frame()
    # drop three weeks of one ISIN to open a gap well beyond a holiday weekend
    mask = ~((df['isin'] == "AAA000000001") &
             (df['date'] >= "2018-02-01") & (df['date'] < "2018-02-21"))
    assert quality_report(df[mask])['max_gap_days'] > MAX_GAP_DAYS


# --------------------------------------------------------------------------
# Integration test: real DB structural invariants (skipped if absent)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(DB_PATH), reason=f"{DB_PATH} not present")
def test_real_db_integrity():
    rep = quality_report(read_long(DB_PATH))
    assert rep['rows'] > 0
    assert rep['duplicates'] == 0
    assert rep['nulls'] == 0
    assert rep['non_positive'] == 0
    assert rep['weekend_rows'] == 0
    assert rep['max_gap_days'] <= MAX_GAP_DAYS
    assert rep['max_abs_return'] < MAX_ABS_RETURN
