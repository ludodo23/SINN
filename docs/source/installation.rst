Installation
============

Prerequisites
-------------

Python 3.10 or higher is recommended.

Create a virtual environment::

    python -m venv .venv

Activate it on Linux/macOS::

    source .venv/bin/activate

On Windows PowerShell::

    .venv\Scripts\Activate.ps1

Install the dependencies::

    pip install -r requirements.txt

Dependencies
------------

The project requires the following packages (see ``requirements.txt``):

- torch — PyTorch deep learning framework
- numpy — numerical computing
- matplotlib — plotting and visualization
- pyvista — OpenFOAM data reading (for ``extract_from_openfoam.py``)
- sphinx — documentation generator
- numpydoc — numpydoc docstring support for Sphinx
- myst-parser — Markdown support for Sphinx
