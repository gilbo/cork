Cork Boolean Library
====================

Welcome to the Cork Boolean/CSG library.  Cork is designed to support Boolean operations between triangle meshes.

Surprisingly, most Boolean/CSG libraries available today (early 2013) are not robust to numerical errors.  Floating-point errors often lead to segmentation faults or produce grossly inaccurate results (e.g. nothing) despite the code being provided .  The few libraries which are robust (e.g. CGAL) require the user to correctly configure the arithmetic settings to ensure robustness.

Cork is designed with the philosophy that you, the user, don't know and don't care about esoteric problems with floating point arithmetic.  You just want a Boolean library with a simple interface, that you can rely on...  Unfortunately since Cork is still in ongoing development, this may be more or less true at the moment.  This code should be very usable for a research project, perhaps slightly less so for use in a product.

Cork was developed by Gilbert Bernstein, a computer scientist who has worked on robust geometric intersections in various projects since 2007.  He's reasonably confident he knows what he's doing. =D

WARNING: Unfortunately, there are a number of known problems with Cork.  I have zero time to work on the library since I'm currently busy working on a thesis. (March 2016) If you would like to pay me large sums of money to improve this library, maybe we should talk.  Otherwise, I'm really quite sorry to say I don't have time.


Installation
============

Dependencies (Mac/Linux)
------------

In order to build Cork on Mac or Linux, you will need Clang 3.1+ and GMP (GNU Multi-Precision arithmetic library).  Eventually, Cork will support GCC.  If you would like more information, or have special system requirements, please e-mail me: I'm much more likely to extend support to a platform if I receive requests.

Mac
---

On OS X 10.8, Clang 3.1+ is the default compiler.  If you are using the Homebrew package manager, I recommend installing GMP that way.

Linux
-----

Clang/LLVM 3.1 and GMP can be installed via your package manager.


Mac/Linux
----

To build the project, type

    make

that's it.


If the build system is unable to find your GMP installation, please edit the paths in file makeConstants.  In general, the project uses a basic makefile.  In the event that you have to do something more complicated to get the library to compile, or if you are unable to get it to compile, please e-mail me or open an issue on GitHub.  Doing so is much more effective than cursing at your computer, and will save other users trouble in the future.


Windows Native
----

Cork uses C++11, so you will need the most recent compiler; Visual Studio 2012 or higher please.  You will also need to install the MPIR arithmetic library into your Visual Studio environment.

Once this is done, you can use the solution and project files in the /win/ subdirectory to build the demo program.  The solution/project is not currently configured to build a DLL.  Please bug me if this is an issue for you.


Cross-Compiling on Unix for Windows
-----------------------------------

Cork can be cross compiled for windows using [mingw-w64](http://mingw-w64.sourceforge.net) and [wclang](https://github.com/tpoechtrager/wclang) which is a clang frontend for mingw. You first need to cross compile the GMP library using mingw-w64 and then:

    make CC=w32-clang CXX=w32-clang++ GMP_INC_DIR=/usr/i686-w64-mingw32/include/ GMP_LIB_DIR=/usr/i686-w64-mingw32/lib/

Tune the `GMP_INC_DIR` and `GMP_LIB_DIR` variables according to where gmp was compiled. It should build a Windows executable under `bin/cork`. You can rename it `cork.exe` and you are set.

Licensing
=========

Cork is licensed under the LGPL with an exception (from QT) to more easily allow use of template code.  In plain English, the following are some guidelines on the use of this code:

*  Unless you also intend to release your project under LGPL/GPL, you must make sure to DYNAMICALLY link against the Cork library.  However, you may include unmodified header files without compromising your proprietary code.

*  If you distribute your code, (publicly or privately, compiled or in source form, for free or commercially) you must (a) acknowledge your use of Cork, (b) include the COPYRIGHT information and (c) either distribute the Cork code or clearly indicate where a copy may be found.

Of course, none of the above supercedes the actual COPYRIGHT.


