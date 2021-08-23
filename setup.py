from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("scripts/edittree_lemmatizer_pipe.pyx")
)