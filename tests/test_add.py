import pandas as pd  # type: ignore
from pdcatcontext import CatContext


def df_init() -> pd.DataFrame:
    df_ = pd.DataFrame(
        {
            "A": ["a", "b", "c"],
            "B": ["d", "e", "f"],
            "C": [1.4, 2.5, 3.6],
            "D": [3, 4, 5],
        }
    )
    return df_


class TestAdd:
    def test_concat_string_columns(self):
        df = df_init()
        df_copy = df.copy()
        df_copy["E"] = df_copy["A"] + df_copy["B"]
        with CatContext(["df"]):
            df["E"] = df["A"] + df["B"]

        assert df.equals(
            df_copy.astype({"A": "category", "B": "category", "E": "category"})
        )
