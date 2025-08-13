"""
Setup script for the prime_mobius package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="prime_mobius",
    version="0.1.0",
    author="Charles Tibedo",
    author_email="charles.tibedo@example.com",
    description="A classical Python implementation of Prime Indexed Möbius Transformation State Space Theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/charlestibedo/prime_mobius",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "sympy>=1.8.0",
        "matplotlib>=3.4.0",
        "scipy>=1.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.6b0",
            "sphinx>=4.0.2",
        ],
    },
)