
from distutils.core import setup, Extension
import numpy as np

cork_module = Extension('_cork',
  sources=['python/cork.i'],
    language="c++",

  # build extension wrapper with c++11 support
  swig_opts=['-c++', '-threads'],
  extra_compile_args=['-std=c++11'],

  libraries=['cork', 'gmp'],
  library_dirs=['lib/'],
  include_dirs=['src', np.get_include()])

setup(
    name='cork',
    version='0.1',
    author='Stephen Dawson-Haggerty',
    description=('Python interface to the cork library'),
    license='GPLv3',
    requires=['numpy'],
    ext_modules=[cork_module],
#     packages = [
#         'cork'
#         ],
#     cmdclass = {
#         'build': BacnetBuild,
#         'build_ext': BacnetBuildExt,
#         },
)
