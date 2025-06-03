# pdcatcontext

`pdcatcontext` is a Python library designed to simplify the management of pandas DataFrames by leveraging the efficiency and versatility of the categorical data type. It provides user-friendly tools to streamline working with categorical data, making it easier to optimize and analyze your datasets.

Categorizing columns of your dataframes makes them lighter and faster to work with. However, there situations that are not support for categorical columns. We can think of concatenation of string type columns, which normally, are as simple as doing `df["A"] + df["B"]`. If you tried to do this kind of operation when `"A"` and `"B` are categorical columns you will get an error as categorical columns can not be add. You can't also concatenate a categorical string type column with a string, so creating key columns as `df["A"] + "-" + df["B"]` is also not allowed with categorical columns.

In this type of situations is where `pdcatcontext` gets really useful, specially, when you frame is so big that is not an option to castback categorical columns to string types just to perform and specific operation.

## QUICKSTART

### How to start using the library

The main object that this library provide is `CatContext`, you can start using it importing from the module after installing with pip. There is also included the `Pointer` class which can be interesting and will be explained later.

`CatContext` is a context manager, this means that is meant to be used in a with block. To start using it you have to provide a list of the string names of the variables that contains your dataframes. An example-we should assume that they are really big dataframes so that the use of categorical dtypes is justify-is given below:

```python
import pandas as pd
from pdcatcontext import CatContext

df1 = pd.DataFrame({"A": [1, 5], "B": ["a", "b"], "C": [2.4, 5.6]})
df2 = pd.DataFrame({"B": ["b", "c"], "D": [7, 8]})

with CatContext(["df1", "df2"]): 
    pass
```

Just by at entering the context, the columns of `df1` and `df2` that are either of object type (string type) or any class of integer (this can be control with the argument `categorize_integers`) type will be converted automatically to categorical columns. Also, columns which have the same name, such as the column `"B"` will be unified. This allows to perform merge operations between the dataframes on that column and benefit from the categorical dtype, because by default, the current pandas dataframe merge operation will cast to object type when performing a merge if the columns are not of the same category. Integer column will be cast back to their original integer type after exiting the context. This behaviour can be change by setting the parameter `cast_back_integers=True`. Inside the context, some of the pandas dataframe methods are overriden to have support for some operations with categorical columns. This will be explained later.

### pdcatcontext vs standard pandas categorical behaviour

Some of the regular pandas dataframe methods are overridden when working inside the `CatContext`. This is done to allow some operations that are not supported by default when working with categorical columns. The following methods are overridden:

- `pd.Series.__add__`: Allows concatenation of categorical string columns with other categorical string columns or with strings. Also, when integers are categorized, allows sum operations between categorical integer columns.

- `pd.Series.apply`: When using `apply` on a categorical column, the function will be efficiently applied since it will only be applied to the unique set of categories and not the entire column, which can significantly speed up operations. However, currently, the dtype of the result will be object type. This override ensures that the resulting column remains categorical in such cases.

- `pd.DataFrame.merge`: One of the biggest issues when mergin dataframes with categorical columns, eventhough the merge is performed on categorical columns with the same name, if the columns dtype is not exactly the same, pandas will cast the column to object type. This override ensures that the merge operation will keep the categorical dtype if the columns are named the same.

- `pd.merge`: Similar to `pd.DataFrame.merge`, this override ensures that the merge operation will keep the categorical dtype if the columns are named the same. This method is call when using `pd.merge(df1, df2)`, while `pd.DataFrame.merge` is called when using `df1.merge(df2)`.

- `pd.DataFrame.groupby`: The default behaviour of `groupby` when there are categorical columns in the columns to group by list is to get all combinations of the categories, which can lead to a very large number of groups. This problem can be natively solved by using the `observed=True` parameter. Inside the context, this parameter is set by default to `True` so you don't have to worry about it. It is possible to control this behaviour by setting the `observed` parameter to `False` in the context manager initialization. We also provide the hability to set the `as_index` parameter to `False` by default at context initialization, so you don't have to set it every time you use `groupby`.

## More to come

This library is still in development and more features will be added in the future. Some of the features that are planned to be added are:

- Be able to use a non-injective map to rename categories in categorical columns.
- Overwrite the `observed` parameter in other pandas dataframe methods that have it, such as `pd.DataFrame.pivot_table`.
- Support for more pandas dataframe methods that are not currently supported with categorical columns.
