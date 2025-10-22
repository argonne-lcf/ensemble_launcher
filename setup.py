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
        "pytest",
        "cloudpickle",
        "pydantic",
        "pyzmq"  
    ],
    extras_require={
        "dragonhpc": ["dragonhpc"],
        "mcp": ["mcp","paramiko"],
    },
    python_requires=">=3.10",
)