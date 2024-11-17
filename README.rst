Thermca
*******

A Python framework to analyse the thermal behaviour of a wide range of
applications, including technical devices, buildings and nature.

What can it do?
===============

It supports analyses of transient thermal behaviour. You can build
models as a network of the following model elements:

- Solid state bodies with geometries of rectangular cuboids and
  cylinders as lumped parameter models with temperature dependent
  material properties;
- Solid state bodies based on the finite element method (FEM); The
  models can be reduced in their degree of freedom using Model Order
  Reduction methods (MOR) for small models and therefore fast
  simulations.
- Simple lumped parameter nodes

Elements can be linked by constant as well as time-varying conductances
and film coefficients. Therefore a comprehensive library of empirical
models for convection, radiation and heat transfer by mass transport
is available. Additional model elements are provided to incorporate
heat loss processes.

The analysis is supported by:

- A domain specific script language for intuitive model definitions and
  result access,
- Automated building of runnable models,
- Simulation with adaptive step size control,
- Domain specific spatial visualisation of model and result data,
- Ability to import FE-meshes.

Installation
============

Thermca is a python based application and requires python 3.10, 3.11 or
3.12.
This project does not make package releases. If you want to install it,
you'll need to download it from GitHub. Thermca has several
dependencies. The preferred method is to install all dependencies as a
pretested Conda virtual environment. Therefore, you need to install the
Anaconda distribution or an lightweight
`Miniconda <https://docs.conda.io/projects/miniconda/en/latest/>`_
variant. The environment information for Thermca is stored in operating
system dependent <operating system and architecture>.lock.YML files.
They can be found in the source root directory. The name of the
environment can be chosen by the -n switch.::

    $ conda env create -f <OS and architecture>_env.lock.yml -n thermca
    $ conda activate thermca

To install Thermca itself in the environment, run pip with its github
address::

    $ pip install git+https://github.com/steffenschroe/Thermca.git --no-dependencies

Using SSH.::

    $ pip install git+ssh://git@github.com/steffenschroe/Thermca --no-dependencies

Package information
-------------------

The following python packages are required:

* numpy: Array calculations
* scipy: Sparse matrix, interpolation and equation solver
* scikit-fem: Finite element routines
* pyvista: 3D-plotting
* sparse: Sparse arrays
* numba: Speed up algorithms written in python
* pandas: Plotting and analysing result data
* meshio: Convert mesh files
* joblib: Cache costly computed numerical objects
* matplotlib: Plotting result data

Optionally, to edit and run models in the JupyterLab environment
following packages are required:

* jupyterlab: As an IDE
* trame: Interactive 3d visualisations in Jupyterlab
* ipywidgets: Interactive 3d visualisations in Jupyterlab

Optionally, to speed up the simulation of FE-based models, the sparse
routines of the Intel Math Kernel Library (MKL) can be used. This
library is sometimes the default implementation of BLAS and LAPACK
routines used by Numpy and Scipy. It is available for Linux x86, MacOS
x86 on Intel processors and Windows x86. The Conda package manager
supports to choose the implementation of BLAS and LAPACK routines during
the installation of Numpy::

    $ conda install conda-forge::numpy "libblas=*=*mkl"

For Apple processors, the Accelerate framework is available::

    $ conda install conda-forge::numpy "libblas=*=*accelerate"

To get access to the MKL sparse routines on Intel processors the
following package has to be installed:

* sparse_dot_mkl

Meshing real world geometries with small features using the common free
mesh tools Gmsh, Netgen, Cgal and Tetgen oftentimes get stuck or
produce a unnecessary high number of cells. Here the robust fTetWild
mesher is recommended. fTetWild is programmed in C++ and has to be
build manually. Supported targets are Linux x86, MacOS x86 and Windows
x86. Meshes for Thermca can be generated using the tetwild.py script
included in the "thermca" source directory. It needs the following
additional Python packages to map the connection surfaces:

* trimesh
* rtree

To build the documentation the following packages have to be installed:

* sphinx
* sphinx_rtd_theme
* nbsphinx

Author
======

My name is Steffen Schroeder and I've created this software as a tool
to help you explore and understand the complex thermal processes that
occur around us. Further, I hope this software will also provide
building blocks other projects can build on.

I wrote this software partly during my work at the Chair of Machine
Tools Development and Adaptive Controls at the TU Dresden and in my
spare time. I built as much as possible on existing Open Source
Software and prior knowledge in the thermal domain. In my opinion,
scientific results should be reproducible, not only for other
researchers, but also for those who later try to apply this knowledge.
In today's world of software-driven science, Open Source is an essential
component for achieving this goal.

Contributors
============
Alexander Galant developed the method to generate thermal FE-systems of
reduced degree of freedom with time varying parameters.

Michael Bauer translated Alexander Galants Matlab-code it into efficient
Python code. He investigated the robustness of the Tetwild mesher and
and developed a method to integrate it into the FEM workflow. Further,
he contributed the initial routines for mesh import and export as well
as functionality to create FE-system matrices with Fenics.

Günter Jungnickel inspired me to write a tool based on thermal lumped
parameter models. The included libraries providing heat transfer and
heat loss models are heavily based on his foundational work.

Maintenance
============
I consider this software to be in feature-complete alpha state. It has
been tested for its main features, but may contain bugs, performance
and stability issues.
Feel free to submit bug reports. I may continue to work on it if I can
find the time to do so. But this project is no longer on the top of my
personal priority list. It would be nice if someone would maintain and
further develop this project.

Acknowledgements
================
I would like to thank professor Steffen Ihlenfeldt and my
colleagues at the Chair of Machine Tools Development and Adaptive
Controls for supporting this work.
The German Science Foundation (DFG) partly funded this software within
the CRC 96 “Thermo-energetic design of machine tools” project T05.

Developer Information
=====================

To install Thermca in development mode, specify the path to the local
source directory::

    $ pip install -e <path/url to Thermca> --no-dependencies

This just links to the given directory. If the sources are moved or 
deleted, importing the package will fail.

Export an working environment to a YAML file::

    $ conda env export > thermca_macos_env.lock.yml


Recommended style guidelines:
-----------------------------

- Google style docstrings
- PEP8 compliance, exception: line length up to 88, doc strings up to
  72
- Double quoted strings if meant to be read by humans, single quoted
  otherwise
- Black code formatter is recommended
- Prefer readability over speed: use temporary names and well named
  functions to document functionality
- Use tau not pi
- Use radius not diameter
- Prefer fully written and meaningful words: "readability counts".
