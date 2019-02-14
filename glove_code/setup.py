from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

NAME = "glove"
VERSION = "0.1"
DESCR = "Python implementation of GloVe"
URL = "http://www.google.com"
REQUIRES = ['numpy', 'cython']

AUTHOR = "Alexandru Tifrea"
EMAIL = "tifreaa@ethz.ch"

LICENSE = "Apache 2.0"

SRC_DIR = "src"
PACKAGES = [SRC_DIR]

ext_1 = Extension(SRC_DIR + "/glove_inner",
                  [SRC_DIR + "/glove_inner.pyx"],
                  extra_compile_args=['-fopenmp'],
                  extra_link_args=['-fopenmp'],
                  libraries=[],
                  include_dirs=[np.get_include()])


EXTENSIONS = [ext_1]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=EXTENSIONS,
          package_data={'': ['*.pyx', '*.pxd', '*.h']},
          include_package_data=True
          )
