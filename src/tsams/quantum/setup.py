from setuptools import setup, find_packages

setup(
    name="cyclotomic_quantum",
    version="0.1.0",
    description="Cyclotomic Field Theory in Quantum Mathematics",
    long_description="""
    This Python package implements the mathematical framework described in the textbook 
    "Cyclotomic Field Theory In Octonionic Topological Braid Theory: Where CNOTs Become Keys". 
    It provides tools for exploring the connections between cyclotomic fields, octonions, 
    braiding structures, and their applications to quantum physics and cosmology.
    
    The core of this framework is the Dedekind cut morphic conductor (value 168 = 2³ × 3 × 7), 
    which serves as a fundamental regulatory principle unifying quantum mechanics, 
    general relativity, and number theory.
    """,
    author="NinjaTech AI",
    author_email="info@ninjatech.ai",
    url="https://github.com/ninjatech-ai/cyclotomic-quantum",
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
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "sympy>=1.8.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
        "qiskit>=0.34.0",  # For quantum computing integrations
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
    },
)