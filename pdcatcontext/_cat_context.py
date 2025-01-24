import pandas as pd  # type: ignore
from pandas.api.types import is_integer_dtype  # type: ignore
from functools import partial
from typing import Any, Optional
from pdcatcontext._pointer import Pointer, PointerName
from pdcatcontext.custom_methods import series_add


def _get_integer_type_map(list_p_df: list[Pointer]) -> dict[int, dict[str, Any]]:
    integer_map = {
        i: {
            col: dtype
            for col, dtype in p_df.dtypes.to_dict().items()
            if is_integer_dtype(dtype)
        }
        for i, p_df in enumerate(list_p_df)
    }
    return integer_map


# region CatContext
class CatContext:
    GLOBALS: Optional[dict[str, Any]] = None

    @classmethod
    def set_globals(cls, globals_: dict[str, Any]) -> None:
        cls.GLOBALS = globals_
        Pointer.set_globals(globals_)

    def __init__(
        self,
        list_p_df: list[PointerName],
        cast_back_integers: bool = True,
        reset_index: bool = True,
    ) -> None:
        if self.GLOBALS is None:
            raise ValueError("Globals not set. Use 'set_globals' method.")

        self._cast_back_integers = cast_back_integers
        self._reeset_index = reset_index
        self._list_p_df: list[Pointer] = [Pointer(p) for p in list_p_df]

        self._integer_dtypes = _get_integer_type_map(self._list_p_df)

        # Default operations that are override
        self._default_series_add = pd.Series.__add__
        self._default_series_apply = pd.Series.apply
        self._default_frame_merge = pd.DataFrame.merge

    def __enter__(self) -> None:
        # Harmonize categories across DataFrames
        self._categorize_strings()
        self._categorize_integers()
        self._unify_categories()

        if self._reeset_index:
            self._reset_index()

        # Override series methods
        pd.Series.__add__ = series_add(self._default_series_add)
        pd.Series.apply = self._series_apply(self._default_series_apply)
        pd.DataFrame.merge = self._frame_merge(self._default_frame_merge)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore overriden operations
        pd.Series.__add__ = self._default_series_add
        pd.Series.apply = self._default_series_apply
        pd.DataFrame.merge = self._default_frame_merge

        # Cast integer columns back to their original type
        if self._cast_back_integers:
            self._recast_integer_types()

    def _reset_index(self):
        """Reset index for all DataFrames"""
        for p_df in self._list_p_df:
            p_df.dereference.reset_index(drop=True, inplace=True)

    def _categorize_strings(self):
        """Cast categorical type to string (object) columns"""
        for p_df in self._list_p_df:
            cat_map = {
                c: "category" for c in p_df.select_dtypes(include="object").columns
            }
            p_df.dereference = p_df.dereference.astype(cat_map)

    def _categorize_integers(self):
        """Cast integer type to integer columns"""
        for p_df in self._list_p_df:
            cat_map = {
                c: "category" for c in p_df.select_dtypes(include="integer").columns
            }
            p_df.dereference = p_df.dereference.astype(cat_map)

    def _unify_categories(self):
        """Unify categories for columns that are named the same"""
        all_columns = set.union(
            *[
                set(p_df.select_dtypes(include="category").columns)
                for p_df in self._list_p_df
            ]
        )
        for col in all_columns:
            values = set.union(
                *[
                    set(p_df.dereference[col].cat.categories)
                    for p_df in self._list_p_df
                    if col in p_df.columns
                ]
            )
            dtype = pd.CategoricalDtype(values, ordered=False)
            for p_df in filter(lambda p_df: col in p_df.columns, self._list_p_df):
                p_df.dereference[col] = p_df.dereference[col].astype(dtype)

    def _recast_integer_types(self):
        """Recast integer type columns to their original integer types after categorization.
        Variables names are used here to point to the dataframe because user might shadowed the variable
        inside the context."""

        for i, p_df in enumerate(self._list_p_df):
            int_map = self._integer_dtypes[i]
            p_df.dereference = p_df.dereference.astype(int_map)

    # region Custom Methods
    def _series_apply(self, default_apply):
        def _custom_apply(self_series, func, *args, **kwargs):
            series_2_return = default_apply(self_series, func, *args, **kwargs)
            if self_series.dtype.name == "category":
                series_2_return = series_2_return.astype("category")

            return series_2_return

        return _custom_apply

    def _frame_merge(self, default_merge):
        def _custom_merge(self_frame, other, *args, **kwargs):
            self._unify_categories()
            return default_merge(self_frame, other, *args, **kwargs)

        return _custom_merge
