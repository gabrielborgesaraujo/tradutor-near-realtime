# -*- mode: python ; coding: utf-8 -*-
"""
AudioTranslate — PyInstaller spec file
Build command: pyinstaller build.spec
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None

# ── Paths ────────────────────────────────────────────────────
VENV_SITE = os.path.join('.venv', 'Lib', 'site-packages')

# ── Hidden imports (módulos que o PyInstaller não detecta) ──
hidden_imports = [
    # PyTorch & CUDA
    *collect_submodules('torch'),
    *collect_submodules('torchaudio'),
    *collect_submodules('torchvision'),
    *collect_submodules('torchcodec'),

    # Whisper
    *collect_submodules('faster_whisper'),
    *collect_submodules('ctranslate2'),

    # Transformers (MarianMT)
    *collect_submodules('transformers'),
    'sentencepiece',

    # Coqui TTS
    *collect_submodules('TTS'),

    # Audio
    'sounddevice',
    '_sounddevice_data',
    'soundfile',

    # Misc
    'encodec',
    'num2words',
    'gruut',
    'pypinyin',
    'jieba',
    'umap',
    'sklearn',
    'scipy',
    'librosa',
    'inflect',
    'unidecode',
    'anyascii',
    'packaging',
    'huggingface_hub',
    'safetensors',
    'tokenizers',
    'regex',
    'filelock',
]

# ── Data files (modelos embutidos, configs, etc) ─────────────
datas = [
    # minha_voz.wav como recurso default
    ('minha_voz.wav', '.'),
]

# Coletar dados dos pacotes
datas += collect_data_files('TTS')
datas += collect_data_files('transformers')
datas += collect_data_files('faster_whisper')
datas += collect_data_files('torch')
datas += collect_data_files('torchaudio')
datas += collect_data_files('torchcodec')
datas += collect_data_files('ctranslate2')
datas += collect_data_files('sounddevice')
datas += collect_data_files('_sounddevice_data')
datas += collect_data_files('num2words')
datas += collect_data_files('gruut')
datas += collect_data_files('encodec')
datas += collect_data_files('librosa')
datas += collect_data_files('inflect')

# ── Binários / DLLs ────────────────────────────────────────
binaries = []
binaries += collect_dynamic_libs('torch')
binaries += collect_dynamic_libs('torchaudio')
binaries += collect_dynamic_libs('torchvision')
binaries += collect_dynamic_libs('torchcodec')
binaries += collect_dynamic_libs('ctranslate2')
binaries += collect_dynamic_libs('sounddevice')

# FFmpeg DLLs
import shutil
_ffmpeg_exe = shutil.which('ffmpeg')
if _ffmpeg_exe:
    _ffmpeg_bin = os.path.dirname(os.path.realpath(_ffmpeg_exe))
    for dll in os.listdir(_ffmpeg_bin):
        if dll.endswith('.dll'):
            binaries.append((os.path.join(_ffmpeg_bin, dll), '.'))

# ── Exclusões (reduzir tamanho) ──────────────────────────────
excludes = [
    'matplotlib',
    'tkinter',
    'IPython',
    'notebook',
    'jupyter',
    'tensorboard',
    'pytest',
    'sphinx',
    'docutils',
]

# ── Analysis ────────────────────────────────────────────────
a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AudioTranslate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,               # UPX não funciona bem com CUDA DLLs
    console=True,             # Manter console para ver logs/debug
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='AudioTranslate',
)
