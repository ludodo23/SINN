Usage
=====

Training
--------

From the project directory, run::

    python train.py

Results are written to ``outputs/`` (or ``runs/`` by default):

- ``sloshing_pinn.pt`` — trained model checkpoint
- ``loss_history.json`` — training loss history

The training script uses the configuration defined in ``config.py``. All
physical and numerical parameters can be adjusted there.

OpenFOAM Data Extraction
------------------------

To extract CFD data from an OpenFOAM case for independent validation::

    python src/extract_from_openfoam.py

This script reads an OpenFOAM case (default: ``../sloshing/sloshing``),
synchronises the fields in time, and saves a compressed ``.npz`` dataset
(default: ``../data/sloshing_data.npz``).

Post-processing
---------------

See ``notebooks/02_postprocess.ipynb`` for visual comparison of OpenFOAM and
PINN results, including animated field comparisons and force time series.
