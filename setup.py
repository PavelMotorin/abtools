#!/usr/bin/env python3
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='abtoaster',
    version='0.1.0',

    description='Tools for A/B test results estimation',
    long_description=long_description,

    url='https://github.com/ivanbagaev/abtoaster',

    author='Ivan Bagaev',
    author_email='ivan.bagaev1993@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'License :: OSI Approved :: MIT License',

    ],

    packages=find_packages(),
    #install_requires=['numpy', 'scipy', 'pymc3', 'matplotlib'],
)
