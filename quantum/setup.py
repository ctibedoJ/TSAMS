"""Setup script for TSAMS package."""

from setuptools import setup, find_packages

setup(
    name="tsams-quantum",
    version="0.1.0",
    description="Quantum applications of the Tibedo Structural Algebraic Modeling System",
    author="Charles Tibedo",
    author_email="charles.tibedo@ninjatech.ai",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib", "qiskit", "cirq", "pennylane"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
