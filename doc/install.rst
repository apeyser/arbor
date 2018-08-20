Installing Arbor
################

Installation of Arbor is done by obtaining the source code and compiling it on
the target system.

This guide starts with an overview of the building process, and the various options
available to customize the build.
The guide then covers installation and running on `HPC clusters <cluster_>`_, followed by a
`troubleshooting guide <troubleshooting_>`_ for common build problems.

.. _install_requirements:

Requirements
============

Minimum Requirements
--------------------

The non distributed (i.e. no MPI) version of Arbor can be compiled on Linux or OS X systems
with very few tools.

.. table:: Required Tools

    =========== ============================================
    Tool        Notes
    =========== ============================================
    Git         To check out the code, minimum version 2.0.
    CMake       To set up the build, minimum version 3.8 (3.9 for MPI).
    compiler    A C++14 compiler. See `compilers <compilers_>`_.
    =========== ============================================

.. _compilers:

Compilers
~~~~~~~~~

Arbor requires a C++ compiler that fully supports C++14.
We recommend using GCC or Clang, for which Arbor has been tested and optimised.

.. table:: Supported Compilers

    =========== ============ ============================================
    Compiler    Min version  Notes
    =========== ============ ============================================
    GCC         6.1.0
    Clang       4.0          Clang 3.8 and later probably work.
    Apple Clang 9
    Intel       17.0.1       Needs GCC 5 or later for standard library.
    =========== ============ ============================================

.. _note_CC:

.. Note::
    The ``CC`` and ``CXX`` environment variables specify which compiler executable
    CMake should use. If these are not set, CMake will attempt to automatically choose a compiler,
    which may be too old to compile Arbor.
    For example, the default compiler chosen below by CMake was GCC 4.8.5 at ``/usr/bin/c++``,
    so the ``CC`` and ``CXX`` variables were used to specify GCC 5.2.0 before calling ``cmake``.

    .. code-block:: bash

        # on this system CMake chooses the following compiler by default
        $ c++ --version
        c++ (GCC) 4.8.5 20150623 (Red Hat 4.8.5-16)

        # check which version of GCC is available
        $ g++ --version
        g++ (GCC) 5.2.0
        Copyright (C) 2015 Free Software Foundation, Inc.

        # set environment variables for compilers
        $ export CC=`which gcc`; export CXX=`which g++`;

        # launch CMake
        # the compiler version and path is given in the CMake output
        $ cmake ..
        -- The C compiler identification is GNU 5.2.0
        -- The CXX compiler identification is GNU 5.2.0
        -- Check for working C compiler: /cm/local/apps/gcc/5.2.0/bin/gcc
        -- Check for working C compiler: /cm/local/apps/gcc/5.2.0/bin/gcc -- works
        ...

.. Note::
    Is is commonly assumed that to get the best performance one should use a vendor-specific
    compiler (e.g. the Intel, Cray or IBM compilers). These compilers are often better at
    auto-vectorizing loops, however for everything else GCC and Clang nearly always generate
    more efficient code.

    The main computational loops in Arbor are generated from
    `NMODL <https://www.neuron.yale.edu/neuron/static/docs/help/neuron/nmodl/nmodl.html>`_.
    The generated code is explicitly vectorised, obviating the need for vendor compilers,
    and we can take advantage of their benefits of GCC and Clang:
    faster compilation times; fewer compiler bugs; and support for recent C++ standards.

.. Note::
    The IBM XL C/C++ compiler for Linux up to version 14 is not supported, owing to unresolved
    compiler issues. We strongly recommend building with GCC or Clang instead on PowerPC
    platforms.

Optional Requirements
---------------------

GPU Support
~~~~~~~~~~~

Arbor has full support for NVIDIA GPUs, for which the NVIDIA CUDA toolkit version 8 is required.

Distributed
~~~~~~~~~~~

Arbor uses MPI to run on HPC cluster systems.
Arbor has been tested on MVAPICH2, OpenMPI, Cray MPI, and IBM MPI.
More information on building with MPI is in the `HPC cluster section <cluster_>`_.

Documentation
~~~~~~~~~~~~~~

To build a local copy of the html documentation that you are reading now, you will need to
install `Sphinx <http://www.sphinx-doc.org/en/master/>`_.

.. _downloading:

Getting the Code
================

The easiest way to acquire the latest version of Arbor is to check the code out from
the `Github repository <https://github.com/eth-cscs/arbor>`_:

.. code-block:: bash

    git clone https://github.com/eth-cscs/arbor.git --recurse-submodules

We recommend using a recursive checkout, because Arbor uses Git submodules for some
of its library dependencies.
The CMake configuration attempts to detect if a required submodule is available, and
will print a helpful warning
or error message if not, but it is up to the user to ensure that all required
submodules are downloaded.

The Git submodules can be updated, or initialized in a project that didn't use a
recursive checkout:

.. code-block:: bash

    git submodule update --init --recursive

You can also point your browser to Arbor's
`Github page <https://github.com/eth-cscs/arbor>`_ and download a zip file.
If you use the zip file, then don't forget to run Git submodule update manually.

.. _building:

Building and Installing Arbor
=============================

Once the Arbor code has been checked out, it can be built by first running CMake to configure the build, then running make.

Below is a simple workflow for: **1)** getting the source; **2)** configuring the build;
**3)** building; **4)** running tests; **5)** install.

For more detailed build configuration options, see the `quick start <quickstart_>`_ guide.

.. code-block:: bash

    # 1) Clone.
    git clone https://github.com/eth-cscs/arbor.git --recurse-submodules
    cd arbor

    # Make a path for building
    mkdir build
    cd build

    # 2) Use CMake to configure the build.
    # By default Arbor builds in release mode, i.e. with optimizations on.
    # Release mode should be used for installing and benchmarking Arbor.
    cmake ..

    # 3) Build Arbor.
    make -j 4

    # 4) Run tests.
    ./test/test.exe
    ./test/global_communication.exe

    # 5) Install (by default, to /usrlocal).
    make install

This will build Arbor in release mode with the `default C++ compiler <note_CC_>`_.

.. _quickstart:

Quick Start: Examples
---------------------

Below are some example of CMake configurations for Arbor. For more detail on individual
CMake parameters and flags, follow links to the more detailed descriptions below.

.. topic:: `Debug <buildtarget_>`_ mode with `assertions <debugging_>`_ enabled.

    If you encounter problems building or running Arbor, compile with these options
    for testing and debugging.

    .. code-block:: bash

        cmake .. -DARB_WITH_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=debug

.. topic:: `Release <buildtarget_>`_ mode (compiler optimizations enabled) with the default
           compiler, optimized for the local `system architecture <architecture_>`_.

    .. code-block:: bash

        cmake .. -DARB_ARCH=native

.. topic:: `Release <buildtarget_>`_ mode with `Clang <compilers_>`_.

    .. code-block:: bash

        export CC=`which clang`
        export CXX=`which clang++`
        cmake ..

.. topic:: `Release <buildtarget_>`_ mode for the `Haswell architecture <architecture_>`_ and `explicit vectorization <vectorize_>`_ of kernels.

    .. code-block:: bash

        cmake .. -DARB_VECTORIZE=ON -DARB_ARCH=haswell

.. topic:: `Release <buildtarget_>`_ mode with `explicit vectorization <vectorize_>`_, targeting the `Broadwell architecture <vectorize_>`_, with support for `P100 GPUs <gpu_>`_, and building with `GCC 5 <compilers_>`_.

    .. code-block:: bash

        export CC=gcc-5
        export CXX=g++-5
        cmake .. -DARB_VECTORIZE=ON -DARB_ARCH=broadwell -DARB_GPU_MODEL=P100

.. topic:: `Release <buildtarget_>`_ mode with `explicit vectorization <vectorize_>`_, optimized for the `local system architecture <architecture_>`_ and `install <install_>`_ in ``/opt/arbor``

    .. code-block:: bash

        cmake .. -DARB_VECTORIZE=ON -DARB_ARCH=native -DCMAKE_INSTALL_PREFIX=/opt/arbor

.. _buildtarget:

Build Target
------------

By default, Arbor is built in release mode, which should be used when installing
or benchmarking Arbor. To compile in debug mode (which in practical terms means
with ``-g -O0`` flags), use the ``CMAKE_BUILD_TYPE`` CMake parameter.

.. code-block:: bash

    cmake -DCMAKE_BUILD_TYPE={debug,release}

..  _architecture:

Architecture
------------

By default, Arbor is built to target whichever architecture is the compiler default,
which often involves a sacrifice of performance for binary portability. The target
architecture can be explicitly set with the ``ARB_ARCH`` configuration option. This
will be used to direct the compiler to use the corresponding instruction sets and
to optimize for that architecture.

When building and installing on the same machine, a good choice for many environments
is to set ``ARB_ARCH`` to ``native``:

.. code-block:: bash

    cmake -DARB_ARCH=native

When deploying on a different machine, one should, for an optimized library, specify
the specific architecture of that machine. The valid values correspond to those given
to the ``-mcpu`` or ``-march`` options for GCC and Clang; the build system will translate
these names to corresponding values for other supported compilers.

Specific recent x86-family Intel CPU architectures include ``broadwell``, ``skylake`` and
``knl``. Complete lists of architecture names can be found in the compiler documentation:
for example GCC `x86 options <https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html>`_,
`PowerPC options <https://gcc.gnu.org/onlinedocs/gcc/RS_002f6000-and-PowerPC-Options.html#RS_002f6000-and-PowerPC-Options>`_,
and `ARM options <https://gcc.gnu.org/onlinedocs/gcc/ARM-Options.html>`_.

..  _vectorize:

Vectorization
-------------

Explicit vectorization of computational kernels can be enabled in Arbor by setting the
``ARB_VECTORIZE`` CMake flag:

.. code-block:: bash

    cmake -DARB_VECTORIZE=ON

With this flag set, the library will use architecture-specific vectorization intrinsics
to implement these kernels. Arbor currently has vectorization support for x86 architectures
with AVX, AVX2 or AVX512 ISA extensions. Enabling the `ARB_VECTORIZE` option for a target
without support in Arbor will give a compilation error.

.. _gpu:

GPU Backend
-----------

Arbor supports NVIDIA GPUs using CUDA. The CUDA back end is enabled by setting the
CMake ``ARB_GPU_MODEL`` option to match the GPU model to target:

.. code-block:: bash

    cmake -DARB_GPU_MODEL={none, K20, K80, P100}

By default ``ARB_GPU_MODEL=none``, and a GPU target must explicitly be set to
build for and run on GPUs.

Depending on the configuration of the system where Arbor is being built, the
C++ compiler may not be able to find the ``cuda.h`` header. The easiest workaround
is to add the path to the include directory containing the header to the
``CPATH`` environment variable before configuring and building Arbor, for
example:

.. code-block:: bash

    export CPATH="/opt/cuda/include:$CPATH"
    cmake -DARB_GPU_MODEL=P100

.. Note::
    The main difference between the Kepler (K20 & K80) and Pascal (P100) GPUs is
    the latter's built-in support for double precision atomics and fewer GPU
    synchronizations when accessing managed memory.

.. _install:

Installation
------------

Arbor can be installed with ``make install`` after configuration. The
installation comprises:

- The static library ``libarbor.a``.
- Public header files.
- The ``modcc`` NMODL compiler if built.
- The HTML documentation if built.

The default install path (``/usr/local``) can be overridden with the standard
``CMAKE_INSTALL_PREFIX`` configuration option.

Provided that Sphinx is available, HTML documentation for Arbor can be built
with ``make html``. Note that documentation is not built by default — if
built, it too will be included in the installation.

Note that the ``modcc`` compiler will not be built by default if the ``ARB_MODCC``
configuration setting is used to specify a different executable for ``modcc``.
While ``modcc`` can be used to translate user-supplied NMODL mechanism
descriptions into C++ and CUDA code for use with Arbor, this generated code
currently relies upon private headers that are not installed.

.. _cluster:

HPC Clusters
============

HPC clusters offer their own unique challenges when compiling and running
software, so we cover some common issues in this section.  If you have problems
on your target system that are not covered here, please make an issue on the
Arbor `Github issues <https://github.com/eth-cscs/arbor/issues>`_ page.
We will do our best to help you directly, and update this guide to help other users.

MPI
---

Arbor uses MPI for distributed systems. By default it is built without MPI support, which
can enabled by setting the ``ARB_WITH_MPI`` configuration flag.
An example of building a 'release' (optimized) version of Arbor with MPI:
is:

.. code-block:: bash

    # set the compiler wrappers
    export CC=`which mpicc`
    export CXX=`which mpicxx`

    # configure with mpi, tbb threading and compiled with optimizations
    cmake .. -DARB_WITH_MPI=ON \           # Use MPI
             -DCMAKE_BUILD_TYPE=release    # Optimizations on

    # run unit tests for global communication on 2 MPI ranks
    mpirun -n 2 ./tests/global_communication.exe

(Note that 'release' build is in fact the default configuration for Arbor.)

The example above sets the ``CC`` and ``CXX`` environment variables to use compiler
wrappers provided by the MPI implementation. While the configuration process
will attempt to find MPI libraries and build options automatically, we recommend
using the supplied MPI compiler wrappers in preference.

.. Note::
    MPI distributions provide **compiler wrappers** for compiling MPI applications.

    In the example above the compiler wrappers for C and C++ called
    ``mpicc`` and ``mpicxx`` respectively. The name of the compiler wrapper
    is dependent on the MPI distribution.

    The wrapper forwards the compilation to a compiler, like GCC, and
    you have to ensure that this compiler is able to compile Arbor. For wrappers
    that call GCC, Intel or Clang compilers, you can pass the ``--version`` flag
    to the wrapper. For example, on a Cray system where the C++ wrapper is called ``CC``:

    .. code-block:: bash

        $ CC --version
        g++ (GCC) 6.2.0 20160822 (Cray Inc.)

Cray Systems
------------

The compiler used by the MPI wrappers is set using a "programming enviroment" module.
The first thing to do is change this module, which by default is set to the Cray
programming environment.
For example, to use the GCC compilers, select the GNU programming enviroment:

.. code-block:: bash

    module swap PrgEnv-cray PrgEnv-gnu

The version of the GCC can then be set by choosing an appropriate gcc module.
In the example below we use ``module avail`` to see which versions of GCC are available,
then choose GCC 7.1.0

.. code-block:: bash

    $ module avail gcc      # see all available gcc versions

    ------------------------- /opt/modulefiles ---------------------------
    gcc/4.9.3    gcc/6.1.0    gcc/7.1.0    gcc/5.3.0(default)    gcc/6.2.0

    $ module swap gcc/7.1.0 # swap gcc 5.3.0 for 7.1.0

    $ CC --version          # test that the wrapper uses gcc 7.1.0
    g++ (GCC) 7.1.0 20170502 (Cray Inc.)

    # set compiler wrappers
    $ export CC=`which cc`
    $ export CXX=`which CC`

Note that the C and C++ compiler wrappers are called ``cc`` and ``CC``
respectively on Cray systems.

CMake detects that it is being run in the Cray programming environment, which makes
our lives a little bit more difficult (CMake sometimes tries a bit too hard to help).
To get CMake to correctly link our code, we need to set the ``CRAYPE_LINK_TYPE``
enviroment variable to ``dynamic``.

.. code-block:: bash

    export CRAYPE_LINK_TYPE=dynamic

Putting it all together, a typicaly workflow to configure the environment and CMake,
then build Arbor is:

.. code-block:: bash

    export CRAYPE_LINK_TYPE=dynamic
    module swap PrgEnv-cray PrgEnv-gnu
    moudle swap gcc/7.1.0
    export CC=`which cc`; export CXX=`which CC`;
    cmake .. -DARB_WITH_MPI=ON           \      # MPI support
             -DCMAKE_BUILD_TYPE=release         # optimized

.. Note::
    If ``CRAYPE_LINK_TYPE`` isn't set, there will be warnings like the following when linking:

    .. code-block:: none

        warning: Using 'dlopen' in statically linked applications requires at runtime
                 the shared libraries from the glibc version used for linking

    Often the library or executable will work, however if a different glibc is loaded,
    Arbor will crash at runtime with obscure errors that are very difficult to debug.


.. _troubleshooting:

Troubleshooting
===============

.. _crosscompiling:

Cross Compiling NMODL
---------------------

Care must be taken when Arbor is compiled on a system with a different
architecture to the target system where Arbor will run. This occurs quite
frequently on HPC systems, for example when building on a login/service node
that has a different architecture to the compute nodes.

.. Note::
    If building Arbor on a laptop or desktop system, i.e. on the same computer that
    you will run Arbor on, cross compilation is not an issue.

.. Note::
    The ``ARB_ARCH`` setting is not applied to the building of ``modcc``.
    On systems where the build node and compute node have different architectures
    within the same family, this may mean that separate compilation of ``modcc``
    is not necessary.

.. Warning::
    ``Illegal instruction`` errors are a sure sign that
    Arbor is running on a system that does not support the architecture it was compiled for.

When cross compiling, we have to take care that the *modcc* compiler, which is
used to convert NMODL to C++/CUDA code, is able to run on the compilation node.

By default, building Arbor will build the ``modcc`` executable from source,
and then use that to build the built-in mechanisms specified in NMODL. This
behaviour can be overridden with the ``ARB_MODCC`` configuration option, for
example:

.. code-block:: bash

   cmake .. -DARB_MODCC=path-to-local-modcc 

Here we will use the example of compiling for Intel KNL on a Cray system, which
has Intel Sandy Bridge CPUs on login nodes that don't support the AVX512
instructions used by KNL.

.. code-block:: bash

    #
    #   Step 1: Build modcc.
    #

    module swap PrgEnv-cray PrgEnv-gnu
    # Important: use GNU compilers directly, not the compiler wrappers,
    # which generate code for KNL, not the login nodes.
    export CC=`which gcc`; export CXX=`which g++`;
    export CRAYPE_LINK_TYPE=dynamic

    # make a path for the modcc build
    mkdir build_modcc
    cd build_modcc

    # configure and make modcc
    cmake ..
    make -j modcc

    #
    #   Step 2: Build Arbor.
    #

    cd ..
    mkdir build; cd build;
    # use the compiler wrappers to build Arbor
    export CC=`which cc`; export CXX=`which CC`;
    cmake .. -DCMAKE_BUILD_TYPE=release           \
             -DARB_WITH_MPI=ON                    \
             -DARB_ARCH=knl                       \
             -DARB_VECTORIZE=ON                   \
             -DARB_MODCC=../build_modcc/bin/modcc


.. Note::
    Cross compilation issues can occur when there are minor differences between login and compute nodes, e.g.
    when the login node has Intel Haswell, and the compute nodes have Intel Broadwell.

    Other systems, such as IBM BGQ, have very different architectures for login and compute nodes.

    If the *modcc* compiler was not compiled for the login node, illegal instruction errors will
    occur when building, e.g.

    .. code-block:: none

        $ make
        ...
        [ 40%] modcc generating: /users/bcumming/arbor_knl/mechanisms/multicore/pas_cpu.hpp
        /bin/sh: line 1: 12735 Illegal instruction     (core dumped) /users/bcumming/arbor_knl/build_modcc/modcc/modcc -t cpu -s\ avx512 -o /users/bcumming/arbor_knl/mechanisms/multicore/pas /users/bcumming/arbor_knl/mechanisms/mod/pas.mod
        mechanisms/CMakeFiles/build_all_mods.dir/build.make:69: recipe for target '../mechanisms/multicore/pas_cpu.hpp' failed

    If you have errors when running the tests or a miniapp, then either the wrong
    ``ARB_ARCH`` target architecture was selected; or you might have forgot to launch on the
    compute node. e.g.:

    .. code-block:: none

        $ ./tests/test.exe
        Illegal instruction (core dumped)

    On the Cray KNL system, ``srun`` is used to launch (it might be ``mpirun``
    or similar on your system):

    .. code-block:: none

        $ srun -n1 -c1 ./tests/test.exe
        [==========] Running 609 tests from 108 test cases.
        [----------] Global test environment set-up.
        [----------] 15 tests from algorithms
        [ RUN      ] algorithms.parallel_sort
        [       OK ] algorithms.parallel_sort (15 ms)
        [ RUN      ] algorithms.sum
        [       OK ] algorithms.sum (0 ms)
        ...


.. _debugging:

Debugging
---------

Sometimes things go wrong: tests fail, simulations give strange results, segmentation
faults occur and exceptions are thrown.

A good first step when things to wrong is to turn on additional assertions that can
catch errors. These are turned off by default (because they slow things down a lot),
and have to be turned on by setting the ``ARB_WITH_ASSERTIONS`` CMake option:

.. code-block:: bash

    cmake -DARB_WITH_ASSERTIONS=ON

.. Note::
    These assertions are in the form of ``arb_assert`` macros inside the code,
    for example:

    .. code-block:: cpp

        void decrement_min_remaining() {
            arb_assert(min_remaining_steps_>0);
            if (!--min_remaining_steps_) {
                compute_min_remaining();
            }
        }

    A failing ``arb_assert`` indicates that an error inside the Arbor
    library, caused either by a logic error in Arbor, or incorrectly checked user input.

    If this occurs, it is highly recommended that you attach the output to the
    `bug report <https://github.com/eth-cscs/arbor/issues>`_ you send to the Arbor developers!


CMake Git Submodule Warnings
----------------------------

When running CMake, warnings like the following indicate that the Git submodules
need to be `updated <downloading_>`_.

.. code-block:: none

    The Git submodule for rtdtheme is not available.
    To check out all submodules use the following commands:
        git submodule init
        git submodule update
    Or download submodules recursively when checking out:
        git clone --recurse-submodules https://github.com/eth-cscs/arbor.git


Wrong Headers for Intel Compiler
------------------------------------

The Intel C++ compiler does not provide its own copy of the C++ standard library,
instead it uses the implementation from GCC. You can see what the default version of
GCC is by ``g++ --version``.

If the Intel compiler uses an old version of the standard library,
errors like the following occur:

.. code-block:: none

    /users/bcumming/arbor_knl/src/util/meta.hpp(127): error: namespace "std" has no member "is_trivially_copyable"
      enable_if_t<std::is_trivially_copyable<T>::value>;

On clusters, a GCC module with a full C++11 implementation of the standard library,
i.e. version 5.1 or later, can be loaded to fix the issue.
