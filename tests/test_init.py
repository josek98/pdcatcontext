import pytest
import pandas as pd  # type: ignore
import numpy as np
from pdcatcontext import CatContext  # Replace with the actual module name


# Fixtures for reusability
@pytest.fixture
def sample_df_with_strings():
    return pd.DataFrame({"A": ["a", "b", "c"], "B": [1, 2, 3], "C": ["x", "y", "z"]})


@pytest.fixture
def sample_df_with_integers():
    return pd.DataFrame(
        {
            "D": np.array([10, 20, 30], dtype="int32"),
            "E": np.array([40, 50, 60], dtype="int64"),
        }
    )


@pytest.fixture
def sample_dfs_for_unification():
    df1 = pd.DataFrame({"Category": ["a", "b"]}, dtype="category")
    df2 = pd.DataFrame({"Category": ["b", "c"]}, dtype="category")
    return df1, df2


# Tests for _categorize_strings
def test_categorize_strings_and_integers(sample_df_with_strings):
    df = sample_df_with_strings
    with CatContext(["df"]):
        # Object columns should be categorized
        assert isinstance(df["A"].dtype, pd.CategoricalDtype)
        assert isinstance(df["C"].dtype, pd.CategoricalDtype)
        # Integer columns should also be categorized
        assert isinstance(df["B"].dtype, pd.CategoricalDtype)


def test_no_string_columns():
    df = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})
    with CatContext(["df"]):
        # Integer columns should still be categorized
        assert isinstance(df["X"].dtype, pd.CategoricalDtype)
        assert isinstance(df["Y"].dtype, pd.CategoricalDtype)


# Tests for _categorize_integers and cast_back_integers
def test_categorize_integers_with_castback(sample_df_with_integers):
    df = sample_df_with_integers
    original_dtypes = df.dtypes.to_dict()

    with CatContext(["df"], cast_back_integers=True):
        # Integers should be categorized during the context
        assert df["D"].dtype == "category"
        assert df["E"].dtype == "category"

    # After exiting, integers should revert to original types
    assert df["D"].dtype == original_dtypes["D"]
    assert df["E"].dtype == original_dtypes["E"]


def test_categorize_integers_without_castback(sample_df_with_integers):
    df = sample_df_with_integers
    with CatContext(["df"], cast_back_integers=False):
        pass  # Context exits without reverting

    # Categories should remain
    assert df["D"].dtype == "category"
    assert df["E"].dtype == "category"


# Tests for _unify_categories (order-agnostic)
def test_unify_categories(sample_dfs_for_unification):
    df1, df2 = sample_dfs_for_unification
    with CatContext(["df1", "df2"]):
        # Check all categories are present (order-agnostic)
        assert set(df1["Category"].cat.categories) == {"a", "b", "c"}
        assert set(df2["Category"].cat.categories) == {"a", "b", "c"}

        # Check values are preserved (not codes)
        assert df1["Category"].tolist() == ["a", "b"]
        assert df2["Category"].tolist() == ["b", "c"]


# Tests for reset_index
def test_reset_index():
    df = pd.DataFrame({"A": [1, 2]}, index=[3, 4])
    with CatContext(["df"], reset_index=True):
        # Index should be reset to default (0, 1)
        assert df.index.tolist() == [0, 1]
    # After context, index remains reset (no undo)
    assert df.index.tolist() == [0, 1]


def test_reset_index_disabled():
    df = pd.DataFrame({"A": [1, 2]}, index=[3, 4])
    original_index = df.index.copy()
    with CatContext(["df"], reset_index=False):
        # Index should remain unchanged
        assert df.index.tolist() == original_index.tolist()
    assert df.index.tolist() == original_index.tolist()


# Edge case: No common columns for unification
def test_unify_categories_no_common_columns():
    df1 = pd.DataFrame({"X": ["a", "b"]}, dtype="category")
    df2 = pd.DataFrame({"Y": ["c", "d"]}, dtype="category")
    with CatContext(["df1", "df2"]):
        # No changes expected
        assert list(df1["X"].cat.categories) == ["a", "b"]
        assert list(df2["Y"].cat.categories) == ["c", "d"]
