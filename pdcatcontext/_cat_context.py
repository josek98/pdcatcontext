from __future__ import annotations

import pandas as pd  # type: ignore
import inspect
from pandas.api.types import is_integer_dtype, union_categoricals  # type: ignore
from typing import Any, Callable
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
    """Context manager for handling categorical data in pandas DataFrames.
    This context manager allows for the automatic categorization of string and integer columns while 
    unifying categories across multiple DataFrames. It also provides methods to add new DataFrames to the context
    and overrides certain pandas operations to ensure consistent behavior across DataFrames.
    """

    def __init__(
        self,
        list_p_df: list[PointerName],
        ignore_columns: list[str] = [],
        categorize_integers: bool = False,
        cast_back_integers: bool = True,
        observed: bool = True,
        as_index: bool = True,
    ) -> None:
        """Initialize the CatContext.

        Parameters
        ----------
        list_p_df : list[PointerName]
            List of DataFrame pointers to include in the context.
        ignore_columns : list[str], optional
            List of columns to ignore during categorization, by default []
        categorize_integers : bool, optional
            Whether to categorize integer columns, by default False
        cast_back_integers : bool, optional
            Whether to cast integer columns back to their original type, by default True
        observed : bool, optional
            Whether to use observed categories for categorical columns, by default True
        as_index : bool, optional
            Whether to use the DataFrame index for categorical columns, by default True

        Raises
        ------
        RuntimeError
            _description_
        """
        current_frame = inspect.currentframe()
        if current_frame is None:
            raise RuntimeError("Failed to retrieve the current frame")

        caller_frame = current_frame.f_back
        caller_locals = caller_frame.f_locals if caller_frame else globals()

        Pointer.set_globals(caller_locals, caller_frame)

        self._ignore_columns = ignore_columns
        self._call_cat_integers = categorize_integers
        self._cast_back_integers = cast_back_integers
        self._observed = observed
        self._as_index = as_index
        self._list_p_df: list[Pointer] = [Pointer(p) for p in list_p_df]

        self._integer_dtypes = _get_integer_type_map(self._list_p_df)

        # Default operations that are override
        self._default_series_add = pd.Series.__add__
        self._default_series_apply = pd.Series.apply
        self._default_frame_merge = pd.DataFrame.merge
        self._default_top_merge = pd.merge
        self._default_frame_groupby = pd.DataFrame.groupby

    def __enter__(self) -> CatContext:
        # Harmonize categories across DataFrames
        self._categorize_strings()
        if self._call_cat_integers:
            self._categorize_integers()

        self._unify_categories()

        # Override series methods
        pd.Series.__add__ = series_add(self._default_series_add)  # type: ignore
        pd.Series.apply = self._series_apply(self._default_series_apply)  # type: ignore
        pd.DataFrame.merge = self._frame_merge(self._default_frame_merge)  # type: ignore
        pd.merge = self._top_merge(self._default_top_merge)  # type: ignore
        pd.DataFrame.groupby = self._frame_groupby()  # type: ignore

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore overriden operations
        pd.Series.__add__ = self._default_series_add
        pd.Series.apply = self._default_series_apply
        pd.DataFrame.merge = self._default_frame_merge
        pd.merge = self._default_top_merge
        pd.DataFrame.groupby = self._default_frame_groupby

        # Cast integer columns back to their original type
        if self._call_cat_integers and self._cast_back_integers:
            self._recast_integer_types()

    # region Public methods
    def add(self, values: PointerName | list[PointerName]) -> None:
        """Add tracking of the given dataframe pointers to the context

        Parameters
        ----------
        values : PointerName | list[PointerName]
            Value or list of the string names of the variables we want to add to the tracking
        """

        _values: list[Pointer]
        if not isinstance(values, list):
            _values = [Pointer(values)]
        else:
            _values = [Pointer(p) for p in values]

        self._list_p_df.extend(_values)
        self._categorize_strings()
        self._categorize_integers()
        self._unify_categories()

        self._integer_dtypes = _get_integer_type_map(self._list_p_df)

    # region Private methods
    def _categorize_strings(self) -> None:
        """Cast categorical type to string (object) columns"""
        for p_df in self._list_p_df:
            cat_map = {
                c: "category"
                for c in p_df.select_dtypes(include="object").columns
                if c not in self._ignore_columns
            }
            p_df.dereference = p_df.dereference.astype(cat_map)

    def _categorize_integers(self) -> None:
        """Cast integer type to integer columns"""
        for p_df in self._list_p_df:
            cat_map = {
                c: "category"
                for c in p_df.select_dtypes(include="integer").columns
                if c not in self._ignore_columns
            }
            p_df.dereference = p_df.dereference.astype(cat_map)

    def _unify_categories(self) -> None:
        """Unify categories for columns that are named the same"""
        all_columns = set.union(
            *[
                set(p_df.select_dtypes(include="category").columns)
                for p_df in self._list_p_df
            ]
        )
        for col in all_columns:
            categories = union_categoricals(
                [
                    p_df.dereference[col]
                    for p_df in self._list_p_df
                    if col in p_df.columns
                ]
            ).categories
            dtype = pd.CategoricalDtype(categories=categories, ordered=False)  # type: ignore
            for p_df in filter(lambda p_df: col in p_df.columns, self._list_p_df):
                p_df.dereference[col] = p_df.dereference[col].astype(dtype)

    def _recast_integer_types(self) -> None:
        """Recast integer type columns to their original integer types after categorization.
        Variables names are used here to point to the dataframe because user might shadowed the variable
        inside the context."""

        for i, p_df in enumerate(self._list_p_df):
            int_map = self._integer_dtypes[i]
            p_df.dereference = p_df.dereference.astype(int_map)

    # region Custom Methods
    def _series_apply(self, default_apply: Callable) -> Callable:
        def _custom_apply(self_series: pd.Series, func: Callable, *args, **kwargs):
            series_2_return = default_apply(self_series, func, *args, **kwargs)
            if self_series.dtype.name == "category":
                series_2_return = series_2_return.astype("category")

            return series_2_return

        return _custom_apply

    def _frame_merge(self, default_merge: Callable) -> Callable:
        def _custom_merge(self_frame: pd.DataFrame, other: object, *args, **kwargs):

            self_match = [p for p in self._list_p_df if self_frame is p.dereference]
            other_match = [p for p in self._list_p_df if other is p.dereference]

            self._categorize_strings()
            if self._call_cat_integers:
                self._categorize_integers()
            self._unify_categories()

            if self_match:
                self_frame = self_match[0].dereference
            if other_match:
                other = other_match[0].dereference

            return default_merge(self_frame, other, *args, **kwargs)

        return _custom_merge

    def _top_merge(self, default_top_merge: Callable) -> Callable:
        def _custom_top_merge(left: pd.DataFrame, right: pd.DataFrame, *args, **kargs):

            left_match = [p for p in self._list_p_df if left is p.dereference]
            right_match = [p for p in self._list_p_df if right is p.dereference]

            self._categorize_strings()
            self._categorize_integers()
            self._unify_categories()

            if left_match:
                left = left_match[0].dereference
            if right_match:
                right = right_match[0].dereference

            return default_top_merge(left, right, *args, **kargs)

        return _custom_top_merge

    def _frame_groupby(self) -> Callable:
        def _custom_groupby(self_frame: pd.DataFrame, *args, **kwargs):
            kwargs.setdefault("observed", self._observed)
            kwargs.setdefault("as_index", self._as_index)
            return self._default_frame_groupby(self_frame, *args, **kwargs)

        return _custom_groupby
