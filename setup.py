import numpy as np
from setuptools import setup, Extension


setup(
    author="Takeshi Ishita",
    author_email="ishitah.takeshi@gmail.com",
    ext_modules=[
        Extension(
            "interpolation",
            sources=["interpolation.pyx", "_bilinear.c"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-g", "-Wall", "-mavx", "-mavx2", "-Ofast"])
    ]
)
