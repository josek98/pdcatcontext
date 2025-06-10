import pandas as pd
import pytest

from pdcatcontext import categorize_and_unify


def test_default_string_unification():
    df1 = pd.DataFrame({"name": pd.Series(["Alice", "Bob"], dtype="string")})
    df2 = pd.DataFrame({"name": pd.Series(["Bob", "Cathy"], dtype="string")})

    categorize_and_unify([df1, df2])

    assert df1["name"].dtype.name == "category"
    assert df2["name"].dtype.name == "category"
    expected = {"Alice", "Bob", "Cathy"}
    assert set(df1["name"].cat.categories) == expected
    assert set(df2["name"].cat.categories) == expected


def test_datetime_unification_unsorted_norange():
    d1 = pd.to_datetime(["2024-01-01", "2024-01-02"])
    d2 = pd.to_datetime(["2024-01-02", "2024-01-03"])
    df1 = pd.DataFrame({"dt": d1})
    df2 = pd.DataFrame({"dt": d2})

    categorize_and_unify(
        [df1, df2], datetimes=True, sorted_datetime=False, full_range_datetime=False
    )

    cats = set(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    assert set(df1["dt"].cat.categories) == cats
    assert df1["dt"].cat.ordered is False
    assert set(df2["dt"].cat.categories) == cats


def test_datetime_full_range_sorted():
    # Two non-overlapping single-day frames
    df1 = pd.DataFrame({"dt": pd.to_datetime(["2023-12-31"])})
    df2 = pd.DataFrame({"dt": pd.to_datetime(["2024-01-02"])})

    categorize_and_unify(
        [df1, df2], datetimes=True, sorted_datetime=True, full_range_datetime=True
    )

    expected = pd.date_range("2023-12-31", "2024-01-02")
    pd.testing.assert_index_equal(df1["dt"].cat.categories, expected)
    pd.testing.assert_index_equal(df2["dt"].cat.categories, expected)
    assert df1["dt"].cat.ordered is True
    assert df2["dt"].cat.ordered is True


def test_mixed_ordered_vs_unordered_categories():
    # DF-1 has an ordered categorical
    df1 = pd.DataFrame(
        {
            "grade": pd.Categorical(
                ["low", "medium"],
                categories=["low", "medium", "high"],
                ordered=True,
            )
        }
    )
    # DF-2 starts as StringDtype
    df2 = pd.DataFrame({"grade": pd.Series(["high", "low"], dtype="string")})

    categorize_and_unify([df1, df2])

    expected = ["low", "medium", "high"]
    assert list(df1["grade"].cat.categories) == expected
    assert list(df2["grade"].cat.categories) == expected
    # Unification always returns *unordered* categories
    assert df1["grade"].cat.ordered is False
