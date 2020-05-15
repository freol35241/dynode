import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()
    
# Parse the requirements-txt file and use for install_requires in pip
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = "dynode",
    version = "0.0.1dev0",
    author = "Fredrik Olsson",
    description = ("""A framework for modellign and simulation of dynamical systems"""),
    url = "https://github.com/freol35241/dynode",
    packages=['dynode'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 4 - Beta"
    ],
    install_requires = required,
)