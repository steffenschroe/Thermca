==========================
Create meshes for FE-parts
==========================

.. toctree::
    :maxdepth: 4

General Workflow
================

* Create geometry with your favourite CAD-tool because typically
  FE-tools like ANSYS are not as convenient to create geometry
* Use STEP file format to transfer the geometry to the FE- or mesh-tool
* Create mesh with coupling surfaces for heat exchange
* Export mesh
* Import mesh in Thermca and use it to create parts

Create geometry with SolidWorks
===============================

Hints
*****

* For meshing in ANSYS: create assembly with multiple parts (bodies) to
  be able to hide bodies in ANSYS and make selection of hidden surfaces
  easier

Split surfaces to create smaller coupling surfaces
**************************************************

Split planar surface

* Create a Sketch on the surface
* Create a Line splitting the surface
* Create a Split Line (Insert > Curve > Split Line), as a one
  directional projection
* For multiple equal part-surfaces: create a Linear Pattern of the Split
  Line with a surface edge line as direction

Split cylindrical surface

* Create Sketch in a plane containing the axis of the cylindrical
  surface
* Create a Line where the sketch plane crosses the cylinder surface
* Create a Split Line (Insert > Curve > Split Line), as a one
  directional projection
* Create a Circular Pattern of the Split Line with a cylinder surface
  edge line as direction

Create geometry with FreeCAD
============================

Split surfaces to create smaller coupling surfaces
**************************************************

Split planar surface

* Select the body face and create a new sketch in Sketcher workbench
* Draw the slicing lines in Sketcher
* Select both, the body and the sketch
* Use Part > Split > Slice to compound
* In Data tab of slice set Mode to Standard

Split cylinder surface

* Create Rectangle in the split plane in Draft workbench
* Make a face from the rectangle: Modification > Upgrade
* Select the rectangle face and create a new sketch in Sketcher
  workbench
* Draw a slicing circle with cylinder radius in Sketcher
* Select both, the body and the sketch
* Use Part > Split > Slice to compound
* In Data tab of slice set Mode to Standard

Create mesh with Salome and/or fTetWild
=======================================

For simple geometries Salome integrated meshing algorithms are
sufficient. On complex real world CAD-geometries these meshing
algorithms oftentimes produce far to high mesh densities or fail to
produce meshes at all. fTetWild can be used in these cases.

Meshing with Salome

* Activate Shaper module
* Create Part: Part > New part
* Import STEP assembly: File > Import > From CAD format ...
* Merge parts: Features > Fuse > select parts while holding Shift key
* Create coupling surfaces:
    * Features > Group > select faces while holding Shift key
    * Hint: if selection does not work after manipulations in the vtk
      graphic window, click in the list box which contain the faces
* Transform Shaper Part to Geometry for meshing: Features > Export to
  GEOM
* Create mesh:
    * Activate the Mesh module
    * Mesh > Create Mesh to define mesh properties
    * Mesh type > Tetrahedral
    * Algorithm > NETGEN 1D-2D-3D or gmesh
    * Select Algorithm and Hypothesis with appropriate parameters
        * sometimes reduction of element size prevents meshing failures
        * Gmsh: change element size via "element size factor"
        * Netgen: change element size via "fineness" and "min size"
    * Mesh > Compute
* Rhight click on mesh in Object Browser and select Export > Med file
  to create a mesh file

Meshing with fTetWild

* Prerequisite is to build fTetWild from source
* Create 2dimensional triangle mesh including the surface groups with
  Salome

    * Same procedure as above but using a 2dimensional meshing algorithm
    * Mesh type > Triangle
    * Algorithm > NETGEN 1D-2D or gmesh
    * The 2dimensional triangle mesh is used as fTetWild input mesh
* Use tetwild.py from thermca directory as user interface
    * tetwild.py generates the 3dimensional mesh including surface
      groups
    * tetwild.py is controlled by command line parameters
    * first two parameters are the file names of the input and output
      mesh
    * use default parameters for fine and following parameters for
      coarse mesh: -l 0.1 -e_abs 0.002 --stop_energy 12.5

Create mesh with GMSH
======================

* Import STEP file to create geometry: File > Open ...
* Add physical group for part body (otherwise the body mesh doesn't get
  exported):

    * In the tree-menu: Modules > Geometry > physical groups > Add
      > Volume
    * Input name for volume
    * Select volume symbol (in center of volume) with LMB in the graphic
      area
    * Press 'e' to end the selection
* Add physical groups for coupling surfaces in the same way:
    * In the tree-menu: Modules > Geometry > physical groups > Add
      > Surface
    * Input name for surface
    * Select surface symbols (dashed grey lines) with LMB in the graphic
      area
    * Hold shift while pressing LMB to deselect surfaces
    * Press 'e' to end the selection
* Create mesh: In the tree-menu: Modules > Mesh > 3D
* Export mesh file: File > Export ... > Save As: Gmsh MSH

Create mesh with ANSYS
======================

* Warning:
    * ANSYS mesh export has bugs and therefore does not work reliably
    * It only works sometimes, if the geometry is made by DesignModeler
    * It does not work, if the geometry is saved by SpaceClaim

Import geometry
***************

* Start WorkBench
* Create Mechanical block
* Import CAD-geometry:
    * Files like .SLDPART containing many geometric elements take very
      long time to import, sometimes the import times out
    * The import of .STEP files is much faster and therefore recommended
    * RMB on Geometry > import to select import file

Create body (parts)
*******************

* Individual sub bodies, like bodies of CAD assembly files, have to be
  merged to make an overall mesh possible
* Small gaps between bodies have to be closed to be able to merge bodies
* By importing several bodies simultaneously, they get connected by
  Contacts automatically
* Make sure to delete automatically created Contacts and set Contact
  settings "Generate Automatic Connection on Refresh" in Mechanical to
  No
* Merge them to one body, with enables an overall mesh in DesignModeler
    * select volumes > RMB > "form new part" (german: markiere Volumen
      > RMT > "Bauteilgruppe erzeugen")
* The body volumes can still be parameterized individually
* Create a body Component to enable automated recognition of the overall
  body mesh in the next workflow steps
* Include all sub bodies in the Component and rename it with "_body" on
  the end of the name

Create coupling surfaces
************************

* Define model coupling surfaces as "Components" that contain surfaces
  of the body geometries
* Create Components in Mechanical not in DesignModeler because
  Mechanical selection tools are better
* Give each Component a name with the ending "_surf" for surface
* Each geometry surface can only be added to one Component, otherwise
  the mesh export fails

Create Mesh
***********

* At the moment only linear tetrahedrons are supported
* To change element geometry in Mechanical insert Method: RMB on Mesh
  > insert > Method
* In the created Method:
    * Assign all sub bodies to the geometry
    * Change Method to Tetrahedron
    * Select Linear function
* Create mesh: RMB on Model > ...







