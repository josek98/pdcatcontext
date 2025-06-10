import pandas as pd
from pandas.api.types import union_categoricals, is_datetime64_any_dtype
from typing import Any

__all__ = ["categorize_and_unify"]


# region Auxiliary functions
def _is_datetime_categories(idx: Any) -> bool:
    """Check if the index represents datetime categories.

    Parameters
    ----------
    idx : Any
        The index to check.

    Returns
    -------
    bool
        True if the index represents datetime categories, False otherwise.
    """

    return is_datetime64_any_dtype(idx) or (
        hasattr(idx, "__len__") and (len(idx) != 0) and isinstance(idx[0], pd.Timestamp)
    )


def _ensure_categorical(s: pd.Series) -> pd.Series:
    """Transform the underlying categorical values to object dtype so string dtype and object dtype
    can be unified. Also ensures different underlying types of integer categories are unified.

    Parameters
    ----------
    s : pd.Series
        The Series to transform.

    Returns
    -------
    pd.Series
        The transformed Series.
    """

    if not isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype("category")

    cats = s.cat.categories
    if not _is_datetime_categories(cats) and cats.dtype != "object":
        s = s.cat.set_categories(cats.astype("object"))
    return s


# region categorize_and_unify function
def categorize_and_unify(
    list_df: list[pd.DataFrame],
    objects: bool = False,
    strings: bool = True,
    integers: bool = False,
    datetimes: bool = True,
    sorted_datetime: bool = False,
    full_range_datetime: bool = False,
) -> None:
    """Transform columns of the DataFrames in `list_df` to categorical type and unify categories
    across all DataFrames. This function modifies the DataFrames in place.

    Parameters
    ----------
    list_df : list[pd.DataFrame]
        The list of DataFrames to transform.
    objects : bool, optional
        Whether to convert object columns to categorical, by default False.
    strings : bool, optional
        Whether to convert string columns to categorical, by default True.
    integers : bool, optional
        Whether to convert integer columns to categorical, by default False.
    datetimes : bool, optional
        Whether to convert datetime columns to categorical, by default True.
    sorted_datetime : bool, optional
        Whether the datetime categories should be sorted, by default False.
    full_range_datetime : bool, optional
        Whether to use the full range of datetime categories, by default False. Full range means
        that the categories will include all dates from the minimum to the maximum date across all
        DataFrames, rather than just the unique dates present in each DataFrame.
    """

    for df in list_df:
        # Transform objects, strings and integers
        type_2_cast: list[str] = []
        if objects:
            type_2_cast.append("object")
        if strings:
            type_2_cast.append("string")
        if integers:
            type_2_cast.append("Int64")

        for col in df.select_dtypes(include=type_2_cast).columns:  # type: ignore
            df[col] = df[col].astype("category")

        # Transform datetimes
        if datetimes:
            for col in df.select_dtypes(include=["datetime"]).columns:
                original = df[col]
                if full_range_datetime:
                    rng = pd.date_range(start=original.min(), end=original.max())
                    df[col] = pd.Categorical(
                        original, categories=rng, ordered=sorted_datetime
                    )
                else:
                    df[col] = pd.Categorical(original, ordered=sorted_datetime)

    # Unify categories across all DataFrames
    cat_cols = set().union(
        *[set(df.select_dtypes(include=["category"]).columns) for df in list_df]
    )

    for col in cat_cols:
        col_series: list[pd.Series] = []
        for df in list_df:
            if col in df.columns:
                df[col] = _ensure_categorical(df[col])
                col_series.append(df[col])

        unified = union_categoricals(
            col_series, ignore_order=True, sort_categories=False
        )
        cats = unified.categories

        # Manage datetime categories
        if datetimes and _is_datetime_categories(cats):
            if full_range_datetime:
                cats = pd.date_range(start=cats.min(), end=cats.max())
            dtype = pd.CategoricalDtype(categories=cats, ordered=sorted_datetime)
        else:
            dtype = pd.CategoricalDtype(categories=cats, ordered=False)

        for df in list_df:
            if col in df.columns:
                df[col] = df[col].astype(dtype)
