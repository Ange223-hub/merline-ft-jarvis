@echo off
title MERLINE - Installation des dependances
cls

echo.
echo ============================================
echo  Installation des dependances MERLINE
echo ============================================
echo.

REM Changer vers le répertoire du script
cd /d "%~dp0"

REM Activer l'environnement virtuel si il existe
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activation de l'environnement virtuel...
    call .venv\Scripts\activate.bat
    echo [INFO] Environnement virtuel active
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activation de l'environnement virtuel...
    call venv\Scripts\activate.bat
    echo [INFO] Environnement virtuel active
) else (
    echo [WARNING] Environnement virtuel non trouve
    echo [INFO] Creation de l'environnement virtuel...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo [INFO] Environnement virtuel cree et active
)

echo.
echo [INFO] Installation des dependances depuis requirements.txt...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERREUR] L'installation a echoue
    echo.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Toutes les dependances ont ete installees!
echo.
pause
