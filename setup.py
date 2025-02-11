from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fsgdm",
    version="1.0.0",
    author="Xianliang Li",
    author_email="yinleung.ley@gmail.com",
    description="The official implementation of the Frequency SGD with Momentum (FSGDM) optimizer in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yinleung/FSGDM",
    packages=find_packages(exclude=("examples")),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.7.0',
    ],
)