#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
  author="Jean-Philippe Fortin, Nick Cullen, Tim Robert-Fitzgerald",
  author_email='fortin946@gmail.com,',
  classifiers=[
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7+',
  ],
  description="ComBat algorithm for harmonizing multi-site imaging data",
  license="MIT license",
  url="https://github.com/Jfortin1/neuroCombat",
  project_urls={
    "Github": "https://github.com/Jfortin1/neuroCombat",
  },
  name='neuroCombat',
  packages=['neuroCombat'],
  version='0.3.0',
  zip_safe=False,
  install_requires=[
    'numpy>=1.16.5',
    'pandas>=1.0.3',
    'scikit-learn>=1.0.0'],
)
