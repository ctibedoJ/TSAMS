from setuptools import setup, find_packages

setup(
    name="tsams-core",
    version="0.1.0",
    description="Core mathematical structures for the Tibedo Structural Algebraic Modeling System",
    author="Charles Tibedo",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
