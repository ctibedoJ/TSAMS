from setuptools import setup, find_packages

setup(
    name="tsams-cryptography",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib", "cryptography", "pycryptodome"],
    author="Charles Tibedo",
    author_email="charles.tibedo@ninjatech.ai",
    description="Cryptographic applications of the Tibedo Structural Algebraic Modeling System (TSAMS)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ctibedoJ/tsams-cryptography",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.8",
)