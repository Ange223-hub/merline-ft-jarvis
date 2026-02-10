@echo off
title MERLINE - Modular Ethical Responsive Local Intelligent Neural Entity
cls

echo.
echo ============================================
echo  MERLINE v1.0
echo  Modular Ethical Responsive Local Intelligent 
echo  Neural Entity
echo ============================================
echo.

REM Changer vers le répertoire du script
cd /d "%~dp0"

REM Activer l'environnement virtuel si il existe
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activation de l'environnement virtuel...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activation de l'environnement virtuel...
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] Environnement virtuel non trouve, utilisation de Python systeme
)

REM Lancer MERLINE
echo [INFO] Lancement de MERLINE...
python launch_merline.py

if errorlevel 1 (
    echo.
    echo [ERREUR] MERLINE ne s'est pas lancee correctement
    echo Verifiez que Python est installe et que toutes les dependencies sont presentes
    echo.
)

pause
