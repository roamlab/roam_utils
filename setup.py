from setuptools import setup, find_packages

setup(
name="roam_utils",
version="0.1.0",
author="ROAM Lab",
description="ROAM Lab's package for utils",
url="https://github.com/roamlab/roam_utils/",
packages=[package for package in find_packages()
                if package.startswith('roam_utils')],
install_requires = [
    'numpy',
    'torch',
    'matplotlib',
    'configparser',
   ]
)
