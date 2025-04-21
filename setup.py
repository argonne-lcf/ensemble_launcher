from setuptools import setup, find_packages

setup(
    name="ensemble_launcher",
    version="0.1.0",
    description="A package for launching ensemble tasks.",
    author="Hari Tummalapalli",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scienceplots",
        "dragonhpc",
        "pytest"
    ],
    python_requires=">=3.6",
)