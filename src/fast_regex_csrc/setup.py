from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "fast_regex",
        ["safe_pcre.cpp"],
        include_dirs=[pybind11.get_include()],
        libraries=["pcre2-8"],
        extra_compile_args=["-std=c++14"],
    ),
]

setup(name="fast_regex", ext_modules=ext_modules)
