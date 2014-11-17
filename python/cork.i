/* This file is part of the Cork library.

 * Cork is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.

 * Cork is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.

 * You should have received a copy 
 * of the GNU Lesser General Public License
 * along with Cork.  If not, see <http://www.gnu.org/licenses/>.
 */
/*
 * @author Stephen Dawson-Haggerty <stevedh@eecs.berkeley.edu>
 */

%module cork
%{
#include <numpy/arrayobject.h>
#include "cork.h"
%}

%init %{
/* do this, or segfault */
import_array();
%}

/* Create typemaps for mapping python types to/from a CorkTriMesh 
 *
 * We allocate temporary PyObjects so that we can use the
 * PyArray_FromAny to accept a lot of different potential kinds of
 * matrixes.  These get cleaned up by the freearg typemap.
 */
%typemap(in) CorkTriMesh (PyObject *a = NULL, PyObject *b = NULL) {
  npy_intp *d1, *d2;

  if (!PyTuple_Check($input) || PyTuple_Size($input) != 2) {
    PyErr_SetString(PyExc_TypeError, "argument must be a tuple of length 2");
    return NULL;
  }
  a = PyTuple_GetItem($input, 0);
  b = PyTuple_GetItem($input, 1);
  if (a == NULL || b == NULL) {
    PyErr_SetString(PyExc_ValueError, "argument must not be none");
    a = b = NULL;
    return NULL;
  }

  /* create the PyArrays */
  a = PyArray_FromAny(a, PyArray_DescrFromType(NPY_UINT), 2, 2, NPY_ARRAY_CARRAY, NULL);
  if (a == NULL) {
    return NULL;
  }
  b = PyArray_FromAny(b, PyArray_DescrFromType(NPY_FLOAT), 2, 2, NPY_ARRAY_CARRAY, NULL);
  if (b == NULL) {
    Py_DECREF(a);
    a = b = NULL;
    return NULL;
  }

  /* check the dimensions and get the heights... */
  d1 = PyArray_DIMS(a);
  d2 = PyArray_DIMS(b);
  if (d1[1] != 3 || d2[1] != 3) {
    PyErr_SetString(PyExc_ValueError, "arrays must be Nx3");
    return NULL;
  }

  /* how to track allocations.. we can't free this since  */
  $1.n_triangles = d1[0];
  $1.n_vertices = d2[0];
  /* very strange -- SWIG generates incorrect code if this A doesn't
     have an argnum. */
  $1.triangles = (uint *)PyArray_DATA(a$argnum);
  $1.vertices = (float *)PyArray_DATA(b);
}

%typemap(freearg) CorkTriMesh {
  if (a$argnum) {
    Py_DECREF(a$argnum);
  }
  if (b$argnum) {
    Py_DECREF(b$argnum);
  }
}

/* allocate a temporary CorkTriMesh on the way in for use as an output
   parameter */
%typemap(in, numinputs=0) CorkTriMesh *OUTPUT (CorkTriMesh temp) {
  memset(&temp, 0, sizeof(temp));
  $1 = &temp;
}

/* build numpy matrices for the return type reusing the memory
   allocated by Cork. */
%typemap(argout) CorkTriMesh *OUTPUT {
  /* return some numpy arrays with the coordinates in there */
  PyObject *a, *b;
  npy_intp dims[2];
  dims[1] = 3;

  dims[0] = temp$argnum.n_triangles;
  a = PyArray_SimpleNewFromData(2, dims, NPY_UINT, temp$argnum.triangles);

  dims[0] = temp$argnum.n_vertices;
  b = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, temp$argnum.vertices);

  $result = Py_BuildValue("(OO)", a, b);
}

/* Export the Cork API */
extern bool isSolid(CorkTriMesh mesh);
extern void computeUnion(CorkTriMesh in0, CorkTriMesh in1, CorkTriMesh *OUTPUT);
extern void computeDifference(CorkTriMesh in0, CorkTriMesh in1, CorkTriMesh *OUTPUT);
extern void computeIntersection(CorkTriMesh in0, CorkTriMesh in1, CorkTriMesh *OUTPUT);
extern void computeSymmetricDifference(
                        CorkTriMesh in0, CorkTriMesh in1, CorkTriMesh *OUTPUT);
extern void resolveIntersections(CorkTriMesh in0, CorkTriMesh in1, CorkTriMesh *OUTPUT);
