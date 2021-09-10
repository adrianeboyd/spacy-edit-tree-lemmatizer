from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

include_dirs = [
]

COMPILER_DIRECTIVES = {
    "language_level": -3,
    "embedsignature": True,
    "annotation_typing": False,
}

MOD_NAMES = [
    "scripts.edittrees",
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

ext_modules = cythonize(ext_modules, compiler_directives=COMPILER_DIRECTIVES)

setup(ext_modules=ext_modules)
