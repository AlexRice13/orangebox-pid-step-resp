#!/usr/bin/env python3
from setuptools import setup, find_packages

import orangebox
import pid_step_response

setup(
    name="orangebox-pid-step-resp",
    version=pid_step_response.__version__,
    packages=find_packages(exclude=["tests", "tests.*"]),
    scripts=["scripts/bb2csv", "scripts/bbsplit", "scripts/bb2gpx"],
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.8.0",
    ],
    extras_require={
        "plotting": ["matplotlib>=3.3.0"],
        "dev": ["pytest>=6.0.0"],
    },
    author="Károly Kiripolszky",
    author_email="karcsi@ekezet.com",
    description="A Cleanflight/Betaflight blackbox log parser with PID step response analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="blackbox cleanflight betaflight pid step-response tuning",
    url="https://github.com/atomgomba/orangebox",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: System :: Archiving :: Compression",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    python_requires=">=3.7"
)
