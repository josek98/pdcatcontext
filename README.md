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

Just by at entering the context, the columns of `df1` and `df2` that are either of object type (string type) or any class of integer type will be converted automatically to categorical columns. Also, columns which have the same name, such as the column `"B"` will be unified. This allows to perform merge operations between the dataframes on that column and benefit from the categorical dtype, because by default, the current pandas dataframe merge operation will cast to object type when performing a merge if the columns are not of the same category. Integer column will be cast back to their original integer type after exiting the context. This behaviour can be change by setting the parameter `cast_back_integers=True`. Inside the context, some of the pandas dataframe methods are overriden to have support for some operations with categorical columns. This will be explained later. 