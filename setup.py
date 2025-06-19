from setuptools import setup, find_packages

setup(
    name="wanda",
    version="0.1.0",
    description="WANDA pruning library",
    author="Sam Orion",
    author_email="your.email@example.com",
    url="https://github.com/Sam-Orion/wanda-python",
    packages=find_packages(include=["wanda", "wanda.*"]),   # include subpackages
    install_requires=[
        "torch",
        "transformers>=4.28.0",
        "datasets>=2.11.0",
        "accelerate>=0.18.0",
        "numpy",
        "sentencepiece",
        "wandb"
    ],
    entry_points={
        "console_scripts": [
            "wanda=wanda.main:main",
            "wanda-opt=wanda.main_opt:main"
        ],
    },
    include_package_data=True,
    package_data={"wanda": ["*.md", "*.txt"]},
)
