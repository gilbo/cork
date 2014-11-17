"""
This file is part of the Cork library.

Cork is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

Cork is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy 
of the GNU Lesser General Public License
along with Cork.  If not, see <http://www.gnu.org/licenses/>.

"""
"""
The setup.py file, and the associated python/cork.i file provide the
typemaps necessary to call cork from Python.

Before building the Python module, you must have compiled the cork
library; it should be placed in lib/ (the default makefile does this.

The swig interface contains typemaps which compile to a single python
module, named _cork.  It exports the definitions from cork.h, except
for freeCorkTriMesh, which is not necessary.

All of these function use the CorkTriMesh type for both input and
output, which is very convenient.  In Python, this type is represented
as a tuple of of (triangles, vertices).  For input, these can be
anything implementing the python iterator protocol.  If you input
numpy matrices, with triangles represented as a uint32 Nx3 matrix, and
the vertices as a float32 Nx3 matrix, it should be possible to work
without a copy.  

The return values are also represented as numpy matrices.  

Currently, input/output from .off files is not implemented.

@author Stephen Dawson-Haggerty <stevedh@eecs.berkeley.edu>
"""

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
)
