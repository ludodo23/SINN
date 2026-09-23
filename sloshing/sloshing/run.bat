@echo off
setlocal

set "FOAM=C:\Program Files\SimFlow\OpenFOAM-2212"

set "FOAM_ETC=%FOAM%\etc"
set "FOAM_APPBIN=%FOAM%\bin"
set "FOAM_LIBBIN=%FOAM%\lib"

REM DLLs et executables
REM IMPORTANT : dummy avant msmpi pour le calcul sequentiel
set "PATH=%FOAM%\bin;%FOAM%\lib\dummy;%FOAM%\lib;%PATH%"

echo ================================
echo OpenFOAM 2212 - SINN Sloshing
echo ================================
echo.


echo [1] Nettoyage des anciens temps...
foamListTimes.exe -rm -noZero

if errorlevel 1 (
    echo ERREUR foamListTimes
    pause
    exit /b 1
)

echo.
echo [2] Initialisation...
setFields.exe

if errorlevel 1 (
    echo ERREUR setFields
    pause
    exit /b 1
)

echo.
echo [3/4] Lancement de interFoam...
echo.

interFoam.exe > interFoam.log 2>&1

echo.
echo [4/4] Creation de sloshing.foam...
type nul > sloshing.foam

echo.
echo ================================
echo Simulation terminee
echo ================================
pause