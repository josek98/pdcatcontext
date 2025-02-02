import pandas as pd  # type: ignore
import pytest
from pdcatcontext import CatContext


# region Fixtures
@pytest.fixture
def sample_df() -> pd.DataFrame:
    df_ = pd.DataFrame(
        {
            "A": ["a", "b", "c"],
            "B": ["d", "e", "f"],
            "C": [1.4, 2.5, 3.6],
            "D": [3, 4, 5],
        }
    )
    return df_


# region Tests
def test_concat_string_columns(sample_df):
    df = sample_df
    df_copy = df.copy()
    df_copy["E"] = df_copy["A"] + df_copy["B"]
    with CatContext(["df"]):
        df["E"] = df["A"] + df["B"]

    assert df.equals(
        df_copy.astype({"A": "category", "B": "category", "E": "category"})
    )


def test_concat_string_column_with_string(sample_df):
    df = sample_df
    df_copy = df.copy()
    df_copy["E"] = df_copy["B"] + "testing"
    with CatContext(["df"]):
        df["E"] = df["B"] + "testing"

    assert df.equals(
        df_copy.astype({"A": "category", "B": "category", "E": "category"})
    )
