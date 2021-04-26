"""
setup file

This file allows the setup of the c extension.
The imported extension contains the K-means algorithm.
"""

from setuptools import setup, Extension

setup(name='mykmeanssp',
      version='1.0',
      description='the c implementation of kmeans step 2',
      ext_modules=[Extension('mykmeanssp', sources=['matoperations.c','kmeans.c'])])
