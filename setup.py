"""Setup script for RED-DiffEq package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="red-diffeq",
    version="1.0.0",
    author="RED-DiffEq Contributors",
    description="Regularization by Denoising Diffusion Models for Solving Inverse PDE Problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/red-diffeq",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "red-diffeq-inversion=scripts.run_inversion:main",
        ],
    },
)

