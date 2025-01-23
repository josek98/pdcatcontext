import pandas as pd  # type: ignore
from pandas.api.types import is_integer_dtype  # type: ignore
from typing import Callable


def _series_add(
    self_series: pd.Series, other: object, default_add: Callable, *args, **kwargs
) -> pd.Series:
    if self_series.dtype.name != "category":
        return default_add(self_series, other, *args, **kwargs)
    
    # Case adding category with string
    if isinstance(other, str):
        self_series_copy = self_series.copy()
        self_values = self_series.dtype.categories.values
        if is_integer_dtype(self_values.dtype):
            map_new = {v: str(v) + other for v in self_values}
        else:
            map_new = {v: v + other for v in self_values}

        self_series_copy = self_series_copy.cat.rename_categories(map_new)
        return self_series_copy
    
    # Case adding categorical strings with same index
    if (
        isinstance(other, pd.Series)
        and self_series.index.equals(other.index)
        and (other.dtype.name == "category")
    ):
        self_name = self_series.name
        self_series_copy = self_series.copy()
        other_copy = other.copy()
        self_series_copy.name = 0
        other_copy.name = 1

        df_combined = pd.concat([self_series_copy, other_copy], axis=1)
        df_combined_reduced = df_combined.drop_duplicates().copy()
        if is_integer_dtype(
            df_combined_reduced[0].dtype.categories.values.dtype
        ):
            if is_integer_dtype(
                df_combined_reduced[1].dtype.categories.values.dtype
            ):
                nc_values = [e1 + e2 for e1, e2 in df_combined_reduced.values]
            else:
                nc_values = [
                    str(e1) + e2 for e1, e2 in df_combined_reduced.values
                ]
        else:
            if is_integer_dtype(
                df_combined_reduced[1].dtype.categories.values.dtype
            ):
                nc_values = [
                    e1 + str(e2) for e1, e2 in df_combined_reduced.values
                ]
            else:
                nc_values = [e1 + e2 for e1, e2 in df_combined_reduced.values]

        df_combined_reduced["NC"] = nc_values
        df_combined_reduced["NC"] = df_combined_reduced["NC"].astype("category")
        df_combined = df_combined.merge(df_combined_reduced)
        self_series_copy = df_combined["NC"]
        self_series_copy.name = self_name
        return self_series_copy

    raise ValueError("Unsupported addition for this type.")
