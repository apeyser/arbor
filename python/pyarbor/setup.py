from setuptools import setup
from distutils.sysconfig import get_python_lib
import glob
import os
import sys

if os.path.exists('readme.rst'):
    print("""The setup.py script should be executed from the build directory.
Please see the file 'readme.rst' for further instructions.""")
    sys.exit(1)


setup(
    name = "pyarbor",
    data_files = [
        (get_python_lib(), glob.glob('*.so')),
        ('bin', ['bin/pyarbor'])
    ],
    author = 'Alexander Peyser',
    description = 'Python Pyarbor for Arbor',
    license = 'BSD',
    keywords = 'arbor pyarbor cmake cython',
    url = 'https://github.com/eth-cscs/arbor',
    #test_require = ['nose'],
    zip_safe = False,
)
