from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sample',
    version='0.0.1',

    description='Tools for A/B test results estimation',
    long_description=long_description,

    url='https://github.com/ivanbagaev/abtools',

    author='Ivan Bagaev',
    author_email='ivanbagaev1993@gmail.com',

    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'License :: OSI Approved :: MIT License',

    ],

    keywords='sample setuptools development',

    packages=['abtools'],
    install_requires=['numpy', 'pymc3', 'seaborn', 'matplotlib'],
)