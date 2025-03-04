import pandas as pd  # type: ignore
import logging
from pandas.api.types import is_integer_dtype  # type: ignore
from typing import Callable

_logger = logging.getLogger(__name__)


## Case self_series is category and other is string
def _case_cat_and_string(self_series: pd.Series, other: str) -> pd.Series:
    self_series_copy = self_series.copy()
    self_values = self_series.dtype.categories.values
    if is_integer_dtype(self_values.dtype):
        map_new = {v: str(v) + other for v in self_values}
    else:
        map_new = {v: v + other for v in self_values}

    self_series_copy = self_series_copy.cat.rename_categories(map_new)
    return self_series_copy


## Case self_series is category and other is categorical series with same index
def _case_cat_cat_index(self_series: pd.Series, other: pd.Series) -> pd.Series:
    self_name = self_series.name
    self_series_copy = self_series.copy()
    other_copy = other.copy()
    self_series_copy.name = 0
    other_copy.name = 1

    df_combined = pd.concat([self_series_copy, other_copy], axis=1)
    df_combined_reduced = df_combined.drop_duplicates().copy()
    if is_integer_dtype(df_combined_reduced[0].dtype.categories.values.dtype):
        if is_integer_dtype(df_combined_reduced[1].dtype.categories.values.dtype):
            nc_values = [e1 + e2 for e1, e2 in df_combined_reduced.values]
        else:
            nc_values = [str(e1) + e2 for e1, e2 in df_combined_reduced.values]
    else:
        if is_integer_dtype(df_combined_reduced[1].dtype.categories.values.dtype):
            nc_values = [e1 + str(e2) for e1, e2 in df_combined_reduced.values]
        else:
            nc_values = [e1 + e2 for e1, e2 in df_combined_reduced.values]

    df_combined_reduced["NC"] = nc_values
    df_combined_reduced["NC"] = df_combined_reduced["NC"].astype("category")
    df_combined = df_combined.merge(df_combined_reduced)

    self_series_copy = self_series_copy.astype(df_combined["NC"].dtype)
    self_series_copy[:] = df_combined["NC"].values
    self_series_copy.name = self_name
    return self_series_copy


def _series_add_logic(
    self_series: pd.Series, other: object, default_series_add: Callable
) -> pd.Series:
    if self_series.dtype.name != "category":
        _logger.debug("Series catcontext add with non-categorical type")
        return default_series_add(self_series, other)

    # Case adding category with string
    if isinstance(other, str):
        _logger.debug("Series catcontext add with category and string")
        return _case_cat_and_string(self_series=self_series, other=other)

    # Case adding categorical strings with same index
    if (
        isinstance(other, pd.Series)
        and self_series.index.equals(other.index)
        and (other.dtype.name == "category")
    ):
        _logger.debug("Series catcontext add with category and category")
        return _case_cat_cat_index(self_series=self_series, other=other)

    raise ValueError("Unsupported addition for this type.")


def series_add(default_series_add: Callable, *args, **kwargs) -> Callable:
    def wrapper(self_series: pd.Series, other: object) -> pd.Series:
        return _series_add_logic(
            self_series=self_series, other=other, default_series_add=default_series_add
        )

    return wrapper
