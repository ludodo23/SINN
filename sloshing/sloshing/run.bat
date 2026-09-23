@echo off
setlocal

set "FOAM=C:\Program Files\SimFlow\OpenFOAM-2212"

echo ==========================================
echo        SINN - Sloshing / OpenFOAM 2212
echo ==========================================

echo.
echo [1/4] Suppression des anciens resultats...
"%FOAM%\bin\foamListTimes.exe" -rm -noZero

echo.
echo [2/4] Initialisation des champs avec setFields...
"%FOAM%\bin\setFields.exe"

if errorlevel 1 (
    echo ERREUR : setFields a echoue.
    pause
    exit /b 1
)

echo.
echo [3/4] Lancement de interFoam...
"%FOAM%\bin\interFoam.exe" > interFoam.log 2>&1

echo.
echo [4/4] Creation du fichier sloshing.foam...
type nul > sloshing.foam

echo.
echo ==========================================
echo        Simulation terminee
echo ==========================================
pause