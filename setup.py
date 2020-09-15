"""
Methods to evaluate dataframes with features and metadata
"""

from setuptools import setup
from setuptools import find_packages

setup(
    name="cytominer_eval",
    description="Methods to evaluate profiling dataframes with features and metadata",
    long_description="Profiling experiments result in a profile, or fingerprint, of a biomedical perturbation of cells. This package evaluates the fingeprint.",
    maintainer="Gregory Way",
    maintainer_email="gregory.way@gmail.com",
    url="https://github.com/cytomining/cytominer-eval",
    packages=find_packages(),
    license="BSD 3-Clause License",
    install_requires=["numpy", "pandas", "scikit-learn"],
    python_requires=">=3.5",
    include_package_data=True,
)
