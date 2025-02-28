'''
Created on Jun 3, 2024

@author: Keith Smith
'''
from setuptools import setup, find_packages

from EMLib.emlib import __version__

setup(
    name='emlib',
    version=__version__,

    url='https://github.com/keith/EMLib',
    author='Keith Smith',
    author_email='keith@snakebite.com',

    packages=find_packages(),

    install_requires=[
        'numpy',
        'scipy',
        'statsmodels',
        'plotly',
        'dash',
        'dash_bootstrap_components',
        'dash_daq'
    ],
)
