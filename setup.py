from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

include_dirs = [
    numpy.get_include(),
]


MOD_NAMES = [
    "scripts.edittree",
    "scripts.edittrees",
    "scripts.edittree_lemmatizer_pipe",
]

ext_modules = []
for name in MOD_NAMES:
    mod_path = name.replace(".", "/") + ".pyx"
    ext = Extension(
        name,
        [mod_path],
        language="c++",
        include_dirs=include_dirs,
        extra_compile_args=["-std=c++11"],
    )
    ext_modules.append(ext)

ext_modules = cythonize(ext_modules)

setup(ext_modules=ext_modules)
