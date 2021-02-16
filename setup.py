import pathlib
from setuptools import setup
from setuptools import find_packages

with open("README.md", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

about = {}
with open(pathlib.Path("cytominer_eval/__about__.py")) as fp:
    exec(fp.read(), about)

setup(
    name="cytominer_eval",
    version=about["__version__"],
    description="Methods to evaluate profiling dataframes with features and metadata",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=about["__author__"],
    author_email="gregory.way@gmail.com",
    url="https://github.com/cytomining/cytominer-eval",
    packages=find_packages(),
    license=about["__license__"],
    install_requires=["numpy", "pandas", "scikit-learn"],
    python_requires=">=3.5",
    include_package_data=True,
)
