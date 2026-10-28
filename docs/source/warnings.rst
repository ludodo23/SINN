Scientific Warnings
===================

This project is a prototype, not a quantitatively validated model.

Geometry and fluid properties
-----------------------------

Geometry parameters, fluid properties, tank motion, and initial level must be
checked against the actual OpenFOAM dictionaries.

Tank motion
-----------

The assumed tank motion is::

    x_tank(t) = A * sin(omega * t + phase)

The network predicts the relative velocity in the tank reference frame. The
predicted pressure is the mechanical pressure ``p``, not directly the OpenFOAM
``p_rgh`` field.

Surface tension
---------------

The sign of the surface tension force should be validated on a test case. The
k-epsilon closure is simplified and does not reproduce all corrections and wall
functions of an OpenFOAM implementation.

Boundary conditions
-------------------

Warning: the code currently imposes ``u = v = 0`` on all four edges of the
rectangle, including the top edge. If the top of the domain is an opening, the
sampling must be modified and appropriate boundary conditions applied.

Initial conditions for ``k`` and ``epsilon`` are not yet imposed explicitly.
The loss weights and their scaling also need to be studied.

Pre-simulation checklist
------------------------

Before running a full simulation, verify:

1. Geometry and patches;
2. Amplitude, frequency and phase of the motion;
3. Physical properties;
4. Initial and boundary conditions;
5. Surface tension sign;
6. Frame and pressure consistency during validation.
