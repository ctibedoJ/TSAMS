"""Setup script for TSAMS package."""

from setuptools import setup, find_packages

setup(
    name="tsams-visualization",
    version="0.1.0",
    description="Visualization applications of the Tibedo Structural Algebraic Modeling System",
    author="Charles Tibedo",
    author_email="charles.tibedo@ninjatech.ai",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib", "plotly", "seaborn", "bokeh"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
