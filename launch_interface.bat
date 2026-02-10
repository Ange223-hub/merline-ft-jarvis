@echo off
title MERLINE - Web Interface
cls

echo.
echo ============================================
echo  MERLINE Web Interface
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
    echo [WARNING] Environnement virtuel non trouve
    echo [INFO] Creation de l'environnement virtuel...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [INFO] Installation des dependances...
    python -m pip install --upgrade pip
    python -m pip install flask flask-cors
)

REM Lancer l'interface web
echo [INFO] Lancement de l'interface web...
python merline_interface.py

if errorlevel 1 (
    echo.
    echo [ERREUR] L'interface ne s'est pas lancee correctement
    echo Verifiez que Flask est installe: pip install flask flask-cors
    echo.
)

pause
