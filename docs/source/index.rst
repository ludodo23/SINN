.. Sloshing PINN documentation master file, created by
   sphinx-quickstart on Tue Oct 28 12:00:00 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Sloshing PINN — Modular Physics-Informed Neural Network Project
===============================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   warnings
   api/modules

Objective
---------

Prototype of a physics-informed neural network (PINN) for a two-dimensional
water–air sloshing problem.

The continuous equations are used as training residuals. No OpenFOAM snapshots
are used as training data. OpenFOAM is reserved for independent validation.

Structure
---------

- config.py — physical and numerical parameters.
- model.py — neural network architecture.
- physics.py — mixture properties and derivatives.
- losses.py — physical residuals and losses.
- sampling.py — collocation point sampling.
- train.py — training loop.
- visualization.py — loss and field visualization.
- extract_from_openfoam.py — OpenFOAM data extraction utility.
- notebooks/01_train.ipynb — training in Jupyter.
- notebooks/02_postprocess.ipynb — post-processing.

Installation
------------

Python 3.10 or higher is recommended.

Create a virtual environment::

    python -m venv .venv

Activate it on Linux/macOS::

    source .venv/bin/activate

On Windows PowerShell::

    .venv\Scripts\Activate.ps1

Install the dependencies::

    pip install -r requirements.txt

Launch
------

From the project directory::

    python train.py

Results are written to ``outputs/``:

- sloshing_pinn.pt
- loss_history.json

Scientific warnings
-------------------

This project is a prototype, not a quantitatively validated model.

Geometry parameters, fluid properties, tank motion, and initial level must be
checked against the actual OpenFOAM dictionaries.

The assumed tank motion is::

    x_tank(t) = A * sin(omega * t + phase)

The network predicts the relative velocity in the tank reference frame. The
predicted pressure is the mechanical pressure ``p``, not directly the OpenFOAM
``p_rgh`` field.

The sign of the surface tension force should be validated on a test case. The
k-epsilon closure is simplified and does not reproduce all corrections and wall
functions of an OpenFOAM implementation.

Warning: the code currently imposes ``u = v = 0`` on all four edges of the
rectangle, including the top edge. If the top of the domain is an opening, the
sampling must be modified and appropriate boundary conditions applied.

The initial conditions for ``k`` and ``epsilon`` are not yet imposed explicitly.
The loss weights and their scaling also need to be studied.

Before running a full simulation, check:

1. Geometry and patches;
2. Amplitude, frequency and phase of the motion;
3. Physical properties;
4. Initial and boundary conditions;
5. Surface tension sign;
6. Frame and pressure consistency during validation.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
