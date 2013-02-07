// +-------------------------------------------------------------------------
// | cork.h
// | 
// | Author: Gilbert Bernstein
// +-------------------------------------------------------------------------
// | COPYRIGHT:
// |    Copyright Gilbert Bernstein 2013
// |    See the included COPYRIGHT file for further details.
// |    
// |    This file is part of the Cork library.
// |
// |    Cork is free software: you can redistribute it and/or modify
// |    it under the terms of the GNU Lesser General Public License as
// |    published by the Free Software Foundation, either version 3 of
// |    the License, or (at your option) any later version.
// |
// |    Cork is distributed in the hope that it will be useful,
// |    but WITHOUT ANY WARRANTY; without even the implied warranty of
// |    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// |    GNU Lesser General Public License for more details.
// |
// |    You should have received a copy 
// |    of the GNU Lesser General Public License
// |    along with Cork.  If not, see <http://www.gnu.org/licenses/>.
// +-------------------------------------------------------------------------
#pragma once

#ifndef uint
typedef unsigned int uint;
#endif

// TODO: describe input format here.

// the inputs to a Boolean operation must be "solid":
//  -   closed (aka. watertight;
//                   every edge has an even number of incident triangles)
//  -   non-self-intersecting
//  -   have consistent CCW triangle orientation
// This function will test whether or not a given mesh is solid
bool isSolid(
    uint n_triangles, uint *triangles,
    uint n_vertices, float *vertices
);

// Boolean operations follow
// result = A U B
void computeUnion(
    // input mesh 0
    uint n_triangles_in0, uint *triangles_in0,
    uint n_vertices_in0, float *vertices_in0,
    // input mesh 1
    uint n_triangles_in1, uint *triangles_in1,
    uint n_vertices_in1, float *vertices_in1,
    // output mesh
    uint *n_triangles_out, uint **triangles_out,
    uint *n_vertices_out, float **vertices_out
);

// result = A - B
void computeDifference(
    // input mesh 0
    uint n_triangles_in0, uint *triangles_in0,
    uint n_vertices_in0, float *vertices_in0,
    // input mesh 1
    uint n_triangles_in1, uint *triangles_in1,
    uint n_vertices_in1, float *vertices_in1,
    // output mesh
    uint *n_triangles_out, uint **triangles_out,
    uint *n_vertices_out, float **vertices_out
);

// result = A ^ B
void computeIntersection(
    // input mesh 0
    uint n_triangles_in0, uint *triangles_in0,
    uint n_vertices_in0, float *vertices_in0,
    // input mesh 1
    uint n_triangles_in1, uint *triangles_in1,
    uint n_vertices_in1, float *vertices_in1,
    // output mesh
    uint *n_triangles_out, uint **triangles_out,
    uint *n_vertices_out, float **vertices_out
);

// result = A XOR B
void computeSymmetricDifference(
    // input mesh 0
    uint n_triangles_in0, uint *triangles_in0,
    uint n_vertices_in0, float *vertices_in0,
    // input mesh 1
    uint n_triangles_in1, uint *triangles_in1,
    uint n_vertices_in1, float *vertices_in1,
    // output mesh
    uint *n_triangles_out, uint **triangles_out,
    uint *n_vertices_out, float **vertices_out
);

// An operation which is not a Boolean operation, but is related:
//  No portion of either surface is deleted.  However, the
//  curve of intersection between the two surfaces is made explicit,
//  such that the two surfaces are now connected.
void resolveIntersections(
    // input mesh 0
    uint n_triangles_in0, uint *triangles_in0,
    uint n_vertices_in0, float *vertices_in0,
    // input mesh 1
    uint n_triangles_in1, uint *triangles_in1,
    uint n_vertices_in1, float *vertices_in1,
    // output mesh
    uint *n_triangles_out, uint **triangles_out,
    uint *n_vertices_out, float **vertices_out
);

