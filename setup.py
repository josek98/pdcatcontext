from setuptools import setup, find_packages #type: ignore

with open("README.md", "r",) as f: 
    long_description = f.read()

setup(
    name="pdcatcontext",
    version="0.1.0b",
    description="Easy use of pandas categorical datatype to optimize dataframe operations time and memory usage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="josek98",
    author_email="josemmsscc98@gmail.com",
    url="https://github.com/josek98/pdcatcontext",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"]
    }
)
