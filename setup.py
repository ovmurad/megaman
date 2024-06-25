# Setup script for megaman: scalable manifold learning
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

from Cython.Build import cythonize
from setuptools import find_packages, setup

DESCRIPTION = "megaman: Manifold Learning for Millions of Points"
LONG_DESCRIPTION = """
megaman: Manifold Learning for Millions of Points
=================================================

This repository contains a scalable implementation of several manifold learning
algorithms, making use of FLANN for fast approximate nearest neighbors and
PyAMG, LOBPCG, ARPACK, and other routines for fast matrix decompositions.

For more information, visit https://github.com/mmp2/megaman
"""
NAME = "megaman"
AUTHOR = "Marina Meila"
AUTHOR_EMAIL = "mmp@stat.washington.delete_this.edu"
URL = "https://github.com/mmp2/megaman"
DOWNLOAD_URL = "https://github.com/mmp2/megaman"
LICENSE = "BSD 3"

VERSION = "0.1.0"

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3.12",
]

setup(
    name="megaman",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    version=VERSION,
    license=LICENSE,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    classifiers=CLASSIFIERS,
)
