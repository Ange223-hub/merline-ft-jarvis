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

REM Utiliser Python 3.11 (seule version avec toutes les dépendances)
C:\Users\steph\AppData\Local\Programs\Python\Python311\python.exe "c:\Users\steph\OneDrive\Bureau\orrrggg merline_may-mlx\merline\launch_merline.py"

if errorlevel 1 (
    echo.
    echo [ERREUR] MERLINE ne s'est pas lancee correctement
    echo Verifiez que Python 3.11 est installe et que toutes les dependencies sont presentes
    echo.
)
pause
