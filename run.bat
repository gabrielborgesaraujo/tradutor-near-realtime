@echo off
chcp 65001 >nul
title AudioTranslate
color 0B

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║       AudioTranslate - Tradutor de Voz           ║
echo  ║       PT → EN em Tempo Real (GPU)                ║
echo  ╚══════════════════════════════════════════════════╝
echo.

if not exist .venv (
    echo  ERRO: Ambiente virtual nao encontrado.
    echo  Execute setup.bat primeiro!
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python app.py

if errorlevel 1 (
    echo.
    echo  A aplicacao encerrou com erro. Verifique o log acima.
    pause
)
