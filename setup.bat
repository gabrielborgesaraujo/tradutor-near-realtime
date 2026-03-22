@echo off
chcp 65001 >nul
title AudioTranslate - Instalação
color 0A

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║       AudioTranslate - Instalação Automatica     ║
echo  ║       Tradutor de Voz em Tempo Real (PT → EN)    ║
echo  ╚══════════════════════════════════════════════════╝
echo.

REM ── 1. Verificar Python ──────────────────────────────────
echo [1/5] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERRO: Python nao encontrado!
    echo  Instale o Python 3.11+ em: https://www.python.org/downloads/
    echo  IMPORTANTE: Marque "Add Python to PATH" durante a instalacao.
    echo.
    pause
    exit /b 1
)
python --version
echo        OK!
echo.

REM ── 2. Verificar NVIDIA GPU / CUDA ──────────────────────
echo [2/5] Verificando GPU NVIDIA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERRO: nvidia-smi nao encontrado!
    echo  Voce precisa de uma GPU NVIDIA com drivers atualizados.
    echo  Baixe em: https://www.nvidia.com/drivers
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader') do (
    echo        GPU detectada: %%i
)
echo        OK!
echo.

REM ── 3. Verificar/Instalar FFmpeg ─────────────────────────
echo [3/5] Verificando FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo        FFmpeg nao encontrado. Instalando via winget...
    winget install Gyan.FFmpeg.Shared --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo  ERRO: Falha ao instalar FFmpeg.
        echo  Instale manualmente: winget install Gyan.FFmpeg.Shared
        echo.
        pause
        exit /b 1
    )
    echo        FFmpeg instalado! Reinicie este script para continuar.
    pause
    exit /b 0
)
echo        OK!
echo.

REM ── 4. Criar ambiente virtual ────────────────────────────
echo [4/5] Criando ambiente virtual Python...
if not exist .venv (
    python -m venv .venv
    echo        Ambiente virtual criado.
) else (
    echo        Ambiente virtual ja existe.
)
echo.

REM ── 5. Instalar dependencias ─────────────────────────────
echo [5/5] Instalando dependencias (isso pode levar 10-20 minutos)...
echo        Baixando PyTorch com CUDA 12.6...
echo.

call .venv\Scripts\activate.bat

REM PyTorch + CUDA 12.6 (do index oficial)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

REM Demais dependências
pip install -r requirements.txt

echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║       Instalação concluida com sucesso!          ║
echo  ║                                                  ║
echo  ║  Para iniciar: execute  run.bat                  ║
echo  ║                                                  ║
echo  ║  NOTA: Na primeira execução, os modelos de IA    ║
echo  ║  serao baixados (~4 GB). Isso e normal.          ║
echo  ╚══════════════════════════════════════════════════╝
echo.
pause
