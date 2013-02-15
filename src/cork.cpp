// +-------------------------------------------------------------------------
// | cork.cpp
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
#include "cork.h"

#include "mesh.h"

struct CorkTriangle;

struct CorkVertex :
    public MinimalVertexData,
    public RemeshVertexData,
    public IsctVertexData,
    public BoolVertexData
{
    void merge(const CorkVertex &v0, const CorkVertex &v1) {
        double                              a0 = 0.5;
        if(v0.manifold && !v1.manifold)     a0 = 0.0;
        if(!v0.manifold && v1.manifold)     a0 = 1.0;
        double a1 = 1.0 - a0;
        
        pos         = a0 * v0.pos       + a1 * v1.pos;
    }
    void interpolate(const CorkVertex &v0, const CorkVertex &v1) {
        double a0 = 0.5;
        double a1 = 0.5;
        pos         = a0 * v0.pos       + a1 * v1.pos;
    }
    
    
    void isct(IsctVertEdgeTriInput<CorkVertex,CorkTriangle> input)
    {
        Vec2d       a_e     = Vec2d(1,1)/2.0;
        Vec3d       a_t     = Vec3d(1,1,1)/3.0;
        a_e /= 2.0;
        a_t /= 2.0;
    }
    void isct(IsctVertTriTriTriInput<CorkVertex,CorkTriangle> input)
    {
        Vec3d       a[3];
        for(uint k=0; k<3; k++) {
            a[k]    = Vec3d(1,1,1)/3.0;
            a[k] /= 3.0;
        }
        for(uint i=0; i<3; i++) {
          for(uint j=0; j<3; j++) {
        }}
    }
    void isctInterpolate(const CorkVertex &v0, const CorkVertex &v1) {
        double a0 = len(v1.pos - pos);
        double a1 = len(v0.pos - pos);
        if(a0 + a1 == 0.0) a0 = a1 = 0.5; // safety
        double sum = a0+a1;
        a0 /= sum;
        a1 /= sum;
    }
};

struct CorkTriangle :
    public MinimalTriangleData,
    public RemeshTriangleData,
    public IsctTriangleData,
    public BoolTriangleData
{
    void merge(const CorkTriangle &, const CorkTriangle &) {}
    static void split(CorkTriangle &, CorkTriangle &,
                      const CorkTriangle &) {}
    void move(const CorkTriangle &) {}
    void subdivide(SubdivideTriInput<CorkVertex,CorkTriangle> input)
    {
        bool_alg_data = input.pt->bool_alg_data;
    }
};

//using RawCorkMesh = RawMesh<CorkVertex, CorkTriangle>;
//using CorkMesh = Mesh<CorkVertex, CorkTriangle>;
typedef RawMesh<CorkVertex, CorkTriangle>   RawCorkMesh;
typedef Mesh<CorkVertex, CorkTriangle>      CorkMesh;

void cMesh2CorkMesh(
    uint n_triangles_in, uint *triangles_in,
    uint n_vertices_in, float *vertices_in,
    CorkMesh *mesh_out
) {
    RawCorkMesh raw;
    raw.vertices.resize(n_vertices_in);
    raw.triangles.resize(n_triangles_in);
    if(n_vertices_in == 0 || n_triangles_in == 0) {
        ERROR("empty mesh input to Cork routine.");
        *mesh_out = CorkMesh(raw);
        return;
    }
    
    uint max_ref_idx = 0;
    for(uint i=0; i<n_triangles_in; i++) {
        raw.triangles[i].a = triangles_in[3*i+0];
        raw.triangles[i].b = triangles_in[3*i+1];
        raw.triangles[i].c = triangles_in[3*i+2];
        max_ref_idx = std::max(
                        std::max(max_ref_idx,
                                 triangles_in[3*i+0]),
                        std::max(triangles_in[3*i+1],
                                 triangles_in[3*i+2])
                      );
    }
    if(max_ref_idx > n_vertices_in) {
        ERROR("mesh input to Cork routine has an out of range vertex index.");
        raw.vertices.clear();
        raw.triangles.clear();
        *mesh_out = CorkMesh(raw);
        return;
    }
    
    for(uint i=0; i<n_vertices_in; i++) {
        raw.vertices[i].pos.x = vertices_in[3*i+0];
        raw.vertices[i].pos.y = vertices_in[3*i+1];
        raw.vertices[i].pos.z = vertices_in[3*i+2];
    }
    
    *mesh_out = CorkMesh(raw);
}
void corkMesh2CMesh(
    CorkMesh *mesh_in,
    uint *n_triangles_out, uint **triangles_out,
    uint *n_vertices_out, float **vertices_out
) {
    RawCorkMesh raw = mesh_in->raw();
    
    *n_triangles_out = raw.triangles.size();
    *n_vertices_out  = raw.vertices.size();
    
    *triangles_out = new uint[(*n_triangles_out) * 3];
    *vertices_out  = new float[(*n_vertices_out) * 3];
    
    for(uint i=0; i<*n_triangles_out; i++) {
        (*triangles_out)[3*i+0] = raw.triangles[i].a;
        (*triangles_out)[3*i+1] = raw.triangles[i].b;
        (*triangles_out)[3*i+2] = raw.triangles[i].c;
    }
    
    for(uint i=0; i<*n_vertices_out; i++) {
        (*vertices_out)[3*i+0] = raw.vertices[i].pos.x;
        (*vertices_out)[3*i+1] = raw.vertices[i].pos.y;
        (*vertices_out)[3*i+2] = raw.vertices[i].pos.z;
    }
}


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
) {
    // convert input
    CorkMesh in0, in1;
    cMesh2CorkMesh(
        n_triangles_in0, triangles_in0,
        n_vertices_in0,  vertices_in0,
        &in0);
    cMesh2CorkMesh(
        n_triangles_in1, triangles_in1,
        n_vertices_in1,  vertices_in1,
        &in1);
    
    in0.boolUnion(in1);
    
    // convert output
    corkMesh2CMesh(&in0,
        n_triangles_out, triangles_out,
        n_vertices_out, vertices_out);
}

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
) {
    // convert input
    CorkMesh in0, in1;
    cMesh2CorkMesh(
        n_triangles_in0, triangles_in0,
        n_vertices_in0,  vertices_in0,
        &in0);
    cMesh2CorkMesh(
        n_triangles_in1, triangles_in1,
        n_vertices_in1,  vertices_in1,
        &in1);
    
    in0.boolDiff(in1);
    
    // convert output
    corkMesh2CMesh(&in0,
        n_triangles_out, triangles_out,
        n_vertices_out, vertices_out);
}

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
) {
    // convert input
    CorkMesh in0, in1;
    cMesh2CorkMesh(
        n_triangles_in0, triangles_in0,
        n_vertices_in0,  vertices_in0,
        &in0);
    cMesh2CorkMesh(
        n_triangles_in1, triangles_in1,
        n_vertices_in1,  vertices_in1,
        &in1);
    
    in0.boolIsct(in1);
    
    // convert output
    corkMesh2CMesh(&in0,
        n_triangles_out, triangles_out,
        n_vertices_out, vertices_out);
}

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
) {
    // convert input
    CorkMesh in0, in1;
    cMesh2CorkMesh(
        n_triangles_in0, triangles_in0,
        n_vertices_in0,  vertices_in0,
        &in0);
    cMesh2CorkMesh(
        n_triangles_in1, triangles_in1,
        n_vertices_in1,  vertices_in1,
        &in1);
    
    in0.boolXor(in1);
    
    // convert output
    corkMesh2CMesh(&in0,
        n_triangles_out, triangles_out,
        n_vertices_out, vertices_out);
}

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
) {
    // convert input
    CorkMesh in0, in1;
    cMesh2CorkMesh(
        n_triangles_in0, triangles_in0,
        n_vertices_in0,  vertices_in0,
        &in0);
    cMesh2CorkMesh(
        n_triangles_in1, triangles_in1,
        n_vertices_in1,  vertices_in1,
        &in1);
    
    in0.disjointUnion(in1);
    in0.resolveIntersections();
    
    // convert output
    corkMesh2CMesh(&in0,
        n_triangles_out, triangles_out,
        n_vertices_out, vertices_out);
}

