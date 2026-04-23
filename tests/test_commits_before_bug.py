"""
test_commits_before_bug.py — compute_stats uclu metrik + by_file testleri.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline import commits_before_bug as cbb


def _df(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["file_path", "commit_idx", "is_bug_intro"])


def test_empty_df_returns_zero_defaults():
    out = cbb.compute_stats(_df([]))
    assert out == {
        "mean_commits_to_first_bug":   0.0,
        "median_commits_to_first_bug": 0.0,
        "mean_commits_between_bugs":   0.0,
        "by_file":                     {},
    }


def test_missing_columns_raises():
    df = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(ValueError):
        cbb.compute_stats(df)


def test_no_bugs_anywhere_returns_empty_by_file():
    out = cbb.compute_stats(_df([
        ("a.py", 0, 0), ("a.py", 1, 0),
        ("b.py", 0, 0),
    ]))
    assert out["by_file"] == {}
    assert out["mean_commits_to_first_bug"]   == 0.0
    assert out["median_commits_to_first_bug"] == 0.0
    assert out["mean_commits_between_bugs"]   == 0.0


def test_single_bug_per_file_first_commit_is_recorded():
    out = cbb.compute_stats(_df([
        ("a.py", 0, 0), ("a.py", 3, 1), ("a.py", 5, 0),
        ("b.py", 0, 0), ("b.py", 1, 0), ("b.py", 9, 1),
    ]))
    assert out["by_file"] == {"a.py": 3, "b.py": 9}
    assert out["mean_commits_to_first_bug"]   == 6.0      # (3+9)/2
    assert out["median_commits_to_first_bug"] == 6.0      # ortanca (iki deger icin ortalama)
    assert out["mean_commits_between_bugs"]   == 0.0      # dosya basina tek bug


def test_multiple_bugs_per_file_computes_between():
    out = cbb.compute_stats(_df([
        ("a.py", 1, 1), ("a.py", 4, 1), ("a.py", 7, 1),  # farklar: 3, 3
        ("b.py", 2, 1), ("b.py", 9, 1),                   # fark: 7
    ]))
    assert out["by_file"] == {"a.py": 1, "b.py": 2}
    assert out["mean_commits_between_bugs"] == round((3 + 3 + 7) / 3, 2)


def test_unsorted_input_sorted_before_analysis():
    """Siralanmamis input commit_idx'ye gore siralanmali."""
    out = cbb.compute_stats(_df([
        ("a.py", 5, 1), ("a.py", 1, 1), ("a.py", 3, 0),
    ]))
    # Ilk bug commit_idx=1 olmali (5 degil)
    assert out["by_file"] == {"a.py": 1}
    assert out["mean_commits_between_bugs"] == 4.0   # 5 - 1


def test_median_odd_and_even_counts():
    # 3 dosya, ilk-bug idx: 1,4,9 -> median=4
    out3 = cbb.compute_stats(_df([
        ("a.py", 1, 1), ("b.py", 4, 1), ("c.py", 9, 1),
    ]))
    assert out3["median_commits_to_first_bug"] == 4.0

    # 4 dosya, ilk-bug idx: 1,3,5,9 -> median=(3+5)/2=4
    out4 = cbb.compute_stats(_df([
        ("a.py", 1, 1), ("b.py", 3, 1), ("c.py", 5, 1), ("d.py", 9, 1),
    ]))
    assert out4["median_commits_to_first_bug"] == 4.0
