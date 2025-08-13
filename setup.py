"""
Setup script for the TSAMS package.
"""

from setuptools import setup, find_packages

setup(
    name="tsams",
    version="0.1.0",
    description="Tibedo Structural Algebraic Modeling System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Charles Tibedo",
    author_email="charles.tibedo@example.com",
    url="https://github.com/ctibedoJ/TSAMS",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "sympy",
    ],
)
