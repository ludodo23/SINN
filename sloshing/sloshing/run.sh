#!/bin/bash

# This script runs the sloshing simulation with the specified parameters.
foamListTimes -rm -noZero # remove all time directories except for 0
setFields # Set the initial fields for the simulation (notable: alpha.eau)
interFoam > interFoam.log 2>&1 & # -parallel # Run the interFoam solver in parallel
touch sloshing.foam # Create a .foam file for visualization in ParaView

