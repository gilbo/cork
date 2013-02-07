// +-------------------------------------------------------------------------
// | main.cpp
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

// This file contains a command line program that can be used
// to exercise Cork's functionality without having to write
// any code.

#include "files.h"

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <sstream>
using std::stringstream;
using std::string;

using std::ostream;

#include "cork.h"


void file2cmesh(
    const Files::FileMesh &in,
    uint *n_triangles_out, uint **triangles_out,
    uint *n_vertices_out, float **vertices_out
) {
    *n_vertices_out = in.vertices.size();
    *n_triangles_out = in.triangles.size();
    
    *triangles_out = new uint[(*n_triangles_out) * 3];
    *vertices_out  = new float[(*n_vertices_out) * 3];
    
    for(uint i=0; i<*n_triangles_out; i++) {
        (*triangles_out)[3*i+0] = in.triangles[i].a;
        (*triangles_out)[3*i+1] = in.triangles[i].b;
        (*triangles_out)[3*i+2] = in.triangles[i].c;
    }
    
    for(uint i=0; i<*n_vertices_out; i++) {
        (*vertices_out)[3*i+0] = in.vertices[i].pos.x;
        (*vertices_out)[3*i+1] = in.vertices[i].pos.y;
        (*vertices_out)[3*i+2] = in.vertices[i].pos.z;
    }
}

void cmesh2file(
    uint n_triangles_in, uint *triangles_in,
    uint n_vertices_in, float *vertices_in,
    Files::FileMesh &out
) {
    out.vertices.resize(n_vertices_in);
    out.triangles.resize(n_triangles_in);
    
    for(uint i=0; i<n_triangles_in; i++) {
        out.triangles[i].a = triangles_in[3*i+0];
        out.triangles[i].b = triangles_in[3*i+1];
        out.triangles[i].c = triangles_in[3*i+2];
    }
    
    for(uint i=0; i<n_vertices_in; i++) {
        out.vertices[i].pos.x = vertices_in[3*i+0];
        out.vertices[i].pos.y = vertices_in[3*i+1];
        out.vertices[i].pos.z = vertices_in[3*i+2];
    }
}

void loadMesh(string filename,
    uint *n_triangles_out, uint **triangles_out,
    uint *n_vertices_out, float **vertices_out
) {
    Files::FileMesh filemesh;
    
    if(Files::readTriMesh(filename, &filemesh) > 0) {
        cerr << "Unable to load in " << filename << endl;
        exit(1);
    }
    
    file2cmesh(filemesh,
        n_triangles_out, triangles_out,
        n_vertices_out, vertices_out
    );
}
void saveMesh(string filename,
    uint n_triangles_in, uint *triangles_in,
    uint n_vertices_in, float *vertices_in
) {
    Files::FileMesh filemesh;
    
    cmesh2file(
        n_triangles_in, triangles_in,
        n_vertices_in, vertices_in,
        filemesh
    );
    
    if(Files::writeTriMesh(filename, &filemesh) > 0) {
        cerr << "Unable to write to " << filename << endl;
        exit(1);
    }
}



class CmdList {
public:
    CmdList();
    ~CmdList() {}
    
    void regCmd(
        string name,
        string helptxt,
        std::function< void(std::vector<string>::iterator &,
                            const std::vector<string>::iterator &) > body
    );
    
    void printHelp(ostream &out);
    void runCommands(std::vector<string>::iterator &arg_it,
                     const std::vector<string>::iterator &end_it);
    
private:
    struct Command {
        string name;    // e.g. "show" will be invoked with option "-show"
        string helptxt; // lines to be displayed
        std::function< void(std::vector<string>::iterator &,
                       const std::vector<string>::iterator &) > body;
    };
    std::vector<Command> commands;
};

CmdList::CmdList()
{
    regCmd("help",
    "-help                  show this help message",
    [this](std::vector<string>::iterator &,
           const std::vector<string>::iterator &) {
        printHelp(cout);
        exit(0);
    });
}

void CmdList::regCmd(
    string name,
    string helptxt,
    std::function< void(std::vector<string>::iterator &,
                        const std::vector<string>::iterator &) > body
) {
    Command cmd = {
        name,
        helptxt,
        body
    };
    commands.push_back(cmd);
}

void CmdList::printHelp(ostream &out)
{
    out <<
    "Welcome to Cork.  Usage:" << endl <<
    "  > cork [-command arg0 arg1 ... argn]*" << endl <<
    "for example," << endl <<
    "  > cork -union box0.off box1.off result.off" << endl <<
    "Options:" << endl;
    for(auto &cmd : commands)
        out << cmd.helptxt << endl;
    out << endl;
}

void CmdList::runCommands(std::vector<string>::iterator &arg_it,
                          const std::vector<string>::iterator &end_it)
{
    while(arg_it != end_it) {
        string arg_cmd = *arg_it;
        if(arg_cmd[0] != '-') {
            cerr << arg_cmd << endl;
            cerr << "All commands must begin with '-'" << endl;
            exit(1);
        }
        arg_cmd = arg_cmd.substr(1);
        arg_it++;
        
        bool found = true;
        for(auto &cmd : commands) {
            if(arg_cmd == cmd.name) {
                cmd.body(arg_it, end_it);
                found = true;
                break;
            }
        }
        if(!found) {
            cerr << "Command -" + arg_cmd + " is not recognized" << endl;
            exit(1);
        }
    }
}


std::function< void(
    std::vector<string>::iterator &,
    const std::vector<string>::iterator &
) >
genericBinaryOp(
    std::function< void(
        // input mesh 0
        uint n_triangles_in0, uint *triangles_in0,
        uint n_vertices_in0, float *vertices_in0,
        // input mesh 1
        uint n_triangles_in1, uint *triangles_in1,
        uint n_vertices_in1, float *vertices_in1,
        // output mesh
        uint *n_triangles_out, uint **triangles_out,
        uint *n_vertices_out, float **vertices_out
    ) > binop
) {
    return [binop]
    (std::vector<string>::iterator &args,
     const std::vector<string>::iterator &end) {
        // data...
        uint nTri0, nTri1, nTriOut;
        uint nVert0, nVert1, nVertOut;
        uint *tri0, *tri1, *triOut;
        float *vert0, *vert1, *vertOut;
        
        if(args == end) { cerr << "too few args" << endl; exit(1); }
        loadMesh(*args, &nTri0, &tri0, &nVert0, &vert0);
        args++;
        
        if(args == end) { cerr << "too few args" << endl; exit(1); }
        loadMesh(*args, &nTri1, &tri1, &nVert1, &vert1);
        args++;
        
        binop(
            nTri0, tri0, nVert0, vert0,
            nTri1, tri1, nVert1, vert1,
            &nTriOut, &triOut, &nVertOut, &vertOut
        );
        
        if(args == end) { cerr << "too few args" << endl; exit(1); }
        saveMesh(*args, nTriOut, triOut, nVertOut, vertOut);
        args++;
        
        delete [] tri0;
        delete [] tri1;
        delete [] triOut;
        delete [] vert0;
        delete [] vert1;
        delete [] vertOut;
    };
}


int main(int argc, char *argv[])
{
    initRand(); // that's useful
    
    if(argc < 2) {
        cout << "Please type 'cork -help' for instructions" << endl;
        exit(0);
    }
    
    // store arguments in a standard container
    std::vector<string> args(argc);
    for(uint k=0; k<argc; k++) {
        args[k] = argv[k];
    }
    
    auto arg_it = args.begin();
    // *arg_it is the program name to begin with, so advance!
    arg_it++;
    
    CmdList cmds;
    
    // add cmds
    cmds.regCmd("union",
    "-union in0 in1 out     Compute the Boolean union of in0 and in1,\n"
    "                       and output the result",
    genericBinaryOp(computeUnion));
    cmds.regCmd("diff",
    "-diff in0 in1 out      Compute the Boolean difference of in0 and in1,\n"
    "                       and output the result",
    genericBinaryOp(computeDifference));
    cmds.regCmd("isct",
    "-isct in0 in1 out      Compute the Boolean intersection of in0 and in1,\n"
    "                       and output the result",
    genericBinaryOp(computeIntersection));
    cmds.regCmd("xor",
    "-xor in0 in1 out       Compute the Boolean XOR of in0 and in1,\n"
    "                       and output the result\n"
    "                       (aka. the symmetric difference)\n",
    genericBinaryOp(computeSymmetricDifference));
    cmds.regCmd("resolve",
    "-resolve in0 in1 out   Intersect the two meshes in0 and in1,\n"
    "                       and output the connected mesh with those\n"
    "                       intersections made explicit and connected",
    genericBinaryOp(resolveIntersections));
    
    
    cmds.runCommands(arg_it, args.end());
    
    return 0;
}









