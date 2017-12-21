#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup  # noqa: F811

setup(
    name="hunse_thesis",
    version="0.1.0",
    author="Eric Hunsberger",
    author_email="ehunsber@uwaterloo.ca",
    packages=['hunse_thesis'],
    scripts=[],
    # url="https://github.com/hunse/thesis",
    license="MIT",
    description="Code for my PhD thesis",
    setup_requires=[
        "numpy>=1.7",
    ],
    install_requires=[
        "numpy>=1.7",
    ],
)
