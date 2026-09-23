#!/bin/bash

echo ==========================================
echo        SINN - Sloshing / OpenFOAM 2212
echo ==========================================

# This script runs the sloshing simulation with the specified parameters.
echo [1/4] Suppression des anciens resultats...
foamListTimes -rm -noZero # remove all time directories except for 0

echo [2/4] Initialisation des champs avec setFields...
setFields # Set the initial fields for the simulation (notable: alpha.eau)
echo [3/4] Lancement de interFoam...
interFoam > interFoam.log 2>&1 & # -parallel # Run the interFoam solver in parallel
echo [4/4] Creation du fichier sloshing.foam...
touch sloshing.foam # Create a .foam file for visualization in ParaView

echo ==========================================
echo        Simulation terminee
echo ==========================================

