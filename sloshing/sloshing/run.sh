#!/bin/bash

echo ==========================================
echo        SINN - Sloshing / OpenFOAM 2212
echo ==========================================

# This script runs the sloshing simulation with the specified parameters.
echo [1/5] Suppression des anciens resultats...
foamListTimes -rm -noZero # remove all time directories except for 0

echo [2/5] Initialisation des champs avec setFields...
setFields # Set the initial fields for the simulation (notable: alpha.eau)
echo [3/5] Lancement de interFoam...
interFoam > interFoam.log 2>&1 # -parallel # Run the interFoam solver in parallel
echo [4/5] Creation du fichier sloshing.foam...
touch sloshing.foam # Create a .foam file for visualization in ParaView
echo [5/5] Reinitialisation du champ alpha.eau...
git restore -- 0/alpha.eau # Reinitialize the alpha.eau field to its initial state

echo ==========================================
echo        Simulation terminee
echo ==========================================

