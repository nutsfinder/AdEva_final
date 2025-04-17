# setup.py
from setuptools import setup, find_packages

setup(
    name="AdEva_final",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.2",
        "PyYAML>=6.0",
        "torch>=2.0.1",
    ],
)