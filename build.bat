@echo off
echo ============================================
echo   AudioTranslate - Build .exe
echo ============================================
echo.

REM Ativa o venv
call .venv\Scripts\activate.bat

REM Limpa builds anteriores
echo Limpando builds anteriores...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Roda o PyInstaller
echo.
echo Iniciando PyInstaller (isso pode levar 5-15 minutos)...
echo.
pyinstaller build.spec --noconfirm

echo.
if exist dist\AudioTranslate\AudioTranslate.exe (
    echo ============================================
    echo   BUILD CONCLUIDO COM SUCESSO!
    echo   Executavel em: dist\AudioTranslate\
    echo ============================================
) else (
    echo ============================================
    echo   ERRO NO BUILD - verifique o log acima
    echo ============================================
)

pause
