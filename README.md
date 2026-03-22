# 🎙️ AudioTranslate

**Tradutor de voz em tempo real com clonagem de voz, acelerado por GPU.**

Fale no microfone em qualquer um dos 17 idiomas suportados e ouça a tradução instantânea — sintetizada com a **sua própria voz** — no idioma de destino.

---

## ✨ Funcionalidades

- 🗣️ **Captura de voz em tempo real** com detecção automática de fala (VAD — Silero)
- 🧠 **Transcrição de alta precisão** via Whisper `large-v3-turbo` (faster-whisper, GPU)
- 🌐 **Tradução neural** entre 17 idiomas usando MarianMT (Helsinki-NLP), com roteamento automático de modelos e *pivot* via inglês quando necessário
- 🔊 **Síntese de voz (TTS)** com clonagem — sua voz é preservada no idioma de destino (Coqui XTTS v2)
- 🖥️ **Interface gráfica moderna** em PyQt6 com tema escuro (Catppuccin)
- ⚡ **Pipeline 100% na GPU** (CUDA) para latência mínima

---

## 🌍 Idiomas Suportados

| Código | Idioma       | Código | Idioma     |
|--------|-------------|--------|------------|
| `pt`   | Português   | `nl`   | Nederlands |
| `en`   | English     | `cs`   | Čeština    |
| `es`   | Español     | `ar`   | العربية    |
| `fr`   | Français    | `zh-cn`| 中文       |
| `de`   | Deutsch     | `ja`   | 日本語     |
| `it`   | Italiano    | `hu`   | Magyar     |
| `pl`   | Polski      | `ko`   | 한국어     |
| `tr`   | Türkçe      | `hi`   | हिन्दी    |
| `ru`   | Русский     |        |            |

> Todos os pares de idiomas são suportados. Quando não existe um modelo direto, o sistema faz tradução via pivô pelo inglês automaticamente.

---

## 🏗️ Arquitetura

```
Microfone → Silero VAD → Whisper STT → MarianMT → XTTS v2 TTS → Alto-falante
              (voz?)       (áudio→texto)  (tradução)   (texto→voz clonada)
```

---

## 📋 Pré-requisitos

| Requisito | Detalhes |
|-----------|----------|
| **OS** | Windows 10/11 |
| **Python** | 3.11+ |
| **GPU** | NVIDIA com suporte a CUDA (recomendado: ≥ 8 GB VRAM) |
| **Drivers NVIDIA** | Atualizados ([download](https://www.nvidia.com/drivers)) |
| **FFmpeg** | Disponível no PATH |

---

## 🚀 Instalação

### Automática (recomendado)

Execute o script de instalação que configura tudo automaticamente:

```bash
setup.bat
```

O script irá:
1. Verificar Python, GPU NVIDIA e FFmpeg
2. Instalar FFmpeg via `winget` caso não encontrado
3. Criar um ambiente virtual Python (`.venv`)
4. Instalar PyTorch com CUDA 12.6 e todas as dependências

### Manual

```bash
# Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate

# Instalar PyTorch com CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Instalar demais dependências
pip install -r requirements.txt
```

---

## ▶️ Como Usar

### Executando

```bash
run.bat
```

Ou manualmente:

```bash
.venv\Scripts\activate
python app.py
```

### Passo a passo

1. Aguarde o carregamento dos modelos (barra de progresso na interface)
2. Selecione os **idiomas** de origem e destino
3. Escolha o **microfone** de entrada e o **dispositivo de saída** de áudio
4. (Opcional) Selecione um arquivo `.wav` de referência para a clonagem de voz
5. Clique em **▶ Iniciar** e comece a falar!

> 💡 **Dica:** Na primeira execução, os modelos de IA serão baixados automaticamente (~4 GB). Isso é normal e só acontece uma vez.

---

## 📦 Gerando Executável (.exe)

Para distribuir a aplicação sem precisar de Python instalado:

```bash
build.bat
```

O executável será gerado em `dist/AudioTranslate/`.

---

## 📁 Estrutura do Projeto

```
AudioTranslate/
├── app.py                   # Aplicação principal (GUI + pipeline completo)
├── captura_vad.py           # Script standalone de teste do VAD
├── transcricao_stt.py       # Script standalone de teste STT (Whisper)
├── transcricao_traducao.py  # Script standalone de teste STT + tradução
├── tradutor_realtime.py     # Script standalone do pipeline completo (console)
├── requirements.txt         # Dependências Python
├── setup.bat                # Instalação automática do ambiente
├── run.bat                  # Atalho para executar a aplicação
├── build.bat                # Script de build do executável
├── build.spec               # Configuração do PyInstaller
└── minha_voz.wav            # Arquivo de referência para clonagem de voz
```

---

## 🛠️ Stack Tecnológica

| Componente | Tecnologia |
|------------|-----------|
| **Interface** | PyQt6 (tema Catppuccin Mocha) |
| **VAD** | Silero VAD |
| **STT** | Faster-Whisper (`large-v3-turbo`, CUDA FP16) |
| **Tradução** | MarianMT — Helsinki-NLP (CUDA) |
| **TTS** | Coqui XTTS v2 (CUDA, multilingual, voice cloning) |
| **Áudio** | sounddevice + NumPy |
| **Build** | PyInstaller |

---

## 📄 Licença

Este projeto é de uso pessoal/educacional. As licenças dos modelos de IA utilizados devem ser respeitadas conforme suas respectivas origens (OpenAI Whisper, Helsinki-NLP, Coqui AI).
