from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wanda",
    version="0.1.0",
    author="Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter",
    author_email="mingjies@cs.cmu.edu",
    description="A Simple and Effective Pruning Approach for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/locuslab/wanda",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.28.0",
        "datasets>=2.11.0",
        "accelerate>=0.18.0",
        "numpy",
        "sentencepiece",
        "wandb",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "wanda=wanda.main:main",
            "wanda-opt=wanda.main_opt:main",
        ],
    },
    include_package_data=True,
    package_data={
        "wanda": ["*.md", "*.txt"],
    },
) 