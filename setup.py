import logging
import sys
from io import open
from os import path

import corgihowfsc

try:
    from setuptools import setup, find_packages
except ImportError:
    logging.exception('Please install or upgrade setuptools or pip')
    sys.exit(1)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# with open("requirements.txt", "r") as f:
#     requirements = f.read().split("\n")

setup(
    name='corgihowfsc',
    # version=corgihowfsc.__version__,
    description=' high-order wavefront sensing and control (HOWFSC) for CGI for advanced algorithm devel',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/roman-corgi/corgihowfsc',
    author='Roman Corongraph CPP Team',
    # author_email='',
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(),
    package_data={
        'corgihowfsc': [
            'scripts/*',
        ]
    },
    include_package_data=True,
    python_requires='>=3.7',
    # install_requires=requirements
)
