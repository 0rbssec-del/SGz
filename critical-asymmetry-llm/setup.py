from setuptools import setup, find_packages

setup(
    name="critical-asymmetry-llm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "transformers>=4.35",
        "numpy>=1.24",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.9",
)
