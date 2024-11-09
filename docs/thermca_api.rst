***********
Thermca API
***********

Model building
**************

Model
=====
.. autoclass:: thermca.Model

Point nodes
===========
.. autoclass:: thermca.Node
.. autoclass:: thermca.BoundNode
.. autoclass:: thermca.MatlNode
.. autoclass:: thermca.StatNode

Lumped parameter parts
======================
.. autoclass:: thermca.LPPart
    :members:
.. autoclass:: thermca.Asm
    :members:
.. autoclass:: thermca.Surf
    :members:
.. autoclass:: thermca.Cube
    :members:
.. autoclass:: thermca.Cyl
    :members:
.. autoclass:: thermca.ForceConts
    :members:


FEM-based parts
===============
.. autoclass:: thermca.FEPart
    :members:
.. autoclass:: thermca.Mesh
    :members:

Heat sources
============
.. autoclass:: thermca.HeatSource
.. autofunction:: thermca.sum_heat
.. autoclass:: thermca.FluxSource

Materials
=========
.. autoclass:: thermca.Solid
.. autoclass:: thermca.Fluid
.. autofunction:: thermca.func_to_table

Input variables
===============
.. autoclass:: thermca.Input
    :members:

Links
=====
.. autoclass:: thermca.CondLink
    :members:
.. autoclass:: thermca.FilmLink
    :members:
.. autofunction:: thermca.sum_films
.. autoclass:: thermca.FlowLink
    :members:
.. autofunction:: thermca.curry

Simulation
**********

Network
=======
.. autoclass:: thermca.Network
    :members:

Result Access
*************
.. autoclass:: thermca.Result
    :members:
.. autoclass:: thermca.resultdata.ElementProcessing
    :members:
.. autoclass:: thermca.resultdata.LinkProcessing
    :members: