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
Test that the python bindings produce the same answer as the c program
on the sample input.

For this to work, you must edit src/util/prelude.h and change initRand
to have an srand(0) in it, so we can get the same results from both
runs.

@author Stephen Dawson-Haggerty <stevedh@eecs.berkeley.edu>
"""

import os
import unittest
import subprocess
import numpy as np
import ctypes
import ctypes.util
import _cork

class TestPythonBindings(unittest.TestCase):
    """Compare the output of the cork binary to the python library
    """
    BALL_A = os.path.join(os.path.dirname(__file__), 
                          "../samples/ballA.off")
    BALL_B = os.path.join(os.path.dirname(__file__), 
                          "../samples/ballB.off")

    @staticmethod
    def read_mesh(f):
        fp = open(f, "r")
        fp.readline()
        dims = map(int, fp.readline().split(" "))
        triangles, vertices = [], []
        for i in xrange(0, dims[0]):
            vertices.append(map(float, fp.readline().split(" ") ))
            
        for i in xrange(0, dims[1]):
            triangles.append(map(float, fp.readline().split(" ")[1:] ))
        return triangles, vertices

    def call_cork(self, method):
        args = [os.path.join(os.path.dirname(__file__), '../bin/cork'),
                '-' + method,
                self.BALL_A, self.BALL_B, '/tmp/_temp_mesh.off']
        subprocess.check_call(args)
        return self.read_mesh('/tmp/_temp_mesh.off')

    def assertTriMeshEquals(self, a, b):
        t1, v1 = a
        t2, v2 = b

        # assert these are close
        self.assertTrue((np.sum(np.abs(v1 - v2)) / np.sum(np.abs(v1))) < 1e-6)
        # these should be equal since they're integers to begin with
        self.assertEqual(np.sum(np.abs(t1 - t2)), 0)

    def setUp(self):
        self.ballA = self.read_mesh(self.BALL_A)
        self.ballB = self.read_mesh(self.BALL_B)

        # are we having fun yet?  cork uses srand to seed the crappy
        # PRNG, but doesn't have a way to reset it.  This way we can
        # call it between runs so we get the same answer every time.
        libc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('c'))
        libc.srand(0)

    def testIsSolid(self):
        """Both demo objects are solid
        """
        self.assertTrue(_cork.isSolid(self.ballA))
        self.assertTrue(_cork.isSolid(self.ballB))

    def testUnion(self):
        self.assertTriMeshEquals(_cork.computeUnion(self.ballA, self.ballB),
                                 self.call_cork('union'))

    def testDifference(self):
        self.assertTriMeshEquals(_cork.computeDifference(self.ballA, self.ballB),
                                 self.call_cork('diff'))

    def testIntersection(self):
        self.assertTriMeshEquals(_cork.computeIntersection(self.ballA, self.ballB),
                                 self.call_cork('isct'))

    def testSymmetricDifference(self):
        self.assertTriMeshEquals(_cork.computeSymmetricDifference(self.ballA, self.ballB),
                                 self.call_cork('xor'))

    def testResolve(self):
        self.assertTriMeshEquals(_cork.resolveIntersections(self.ballA, self.ballB),
                                 self.call_cork('resolve'))


if __name__ == '__main__':
    unittest.main()
