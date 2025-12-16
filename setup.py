from setuptools import setup, find_packages

setup(
    name="lightron",
    version="0.1.0",
    description="A lightweight, modern distributed training framework for LLMs",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy",
    ],
)
