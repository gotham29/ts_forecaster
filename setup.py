#!/usr/bin/env python

import os

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()


def locate(*names):
    return os.path.relpath(os.path.join(os.path.dirname(__file__), *names))


SOURCE = locate('ts_source')
PACKAGES_REQUIRED = find_packages(SOURCE)
PACKAGES_REQUIRED = [os.path.join(SOURCE, x) for x in PACKAGES_REQUIRED]
if 'ts_source' not in PACKAGES_REQUIRED:
    PACKAGES_REQUIRED.insert(0, 'ts_source')

print(f"SOURCE = {SOURCE}")
print(f"PACKAGES_REQUIRED...")
for pr in sorted(PACKAGES_REQUIRED):
    print(f"  --> {pr}\n")

INSTALL_REQUIRES = [
    'pandas==1.2.4',
    'setuptools>=34.4.0', # needed for Windows with MSVC
    'pip>=19.3.1',
    'pandas>=1.2.4',
    'PyYAML==6.0',
    'numpy>=1.15',
    'requests',
    'darts'
]

authors = "Sam Heiserman"
author_email = "sheiser1@binghamton.edu"

setup(
    name='ts_source',
    version='0.0.9999999999948',
    description='Timeseries Forecaster - Rapid ML prototyping tool for models forecasting on numeric time series',
    long_description=readme,
    license='MIT',
    author=authors,
    author_email=author_email,
    url='https://github.com/gotham29/ts_forecaster',
    project_urls={
        "Bug Tracker": "https://github.com/gotham29/ts_forecaster/issues"
    },
    packages=PACKAGES_REQUIRED,
    install_requires=INSTALL_REQUIRES,
    keywords=["pypi", "ts_source"],
)
