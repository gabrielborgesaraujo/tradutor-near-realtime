"""
AudioTranslate — Tradutor de Voz em Tempo Real (Multi-idiomas)
Pipeline: Microfone → VAD → Whisper STT → MarianMT → XTTS TTS → Saída
Interface gráfica com PyQt6, processamento 100% na GPU.
Suporta 17 idiomas compatíveis com XTTS v2.
"""

import sys
import os
import shutil
import warnings
import ssl
import queue

# ── Registrar DLLs do FFmpeg antes de qualquer import que use torchcodec ──
# O torchcodec (usado pelo Coqui TTS) precisa das DLLs do FFmpeg acessíveis.
# No Windows, os.add_dll_directory() é necessário para que o ctypes encontre.
if sys.platform == "win32":
    _ffmpeg_exe = shutil.which("ffmpeg")
    if _ffmpeg_exe:
        _ffmpeg_bin = os.path.dirname(os.path.realpath(_ffmpeg_exe))
        os.add_dll_directory(_ffmpeg_bin)
        # Também adiciona ao PATH para cobrir importações indiretas
        os.environ["PATH"] = _ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")
        print(f"[INFO] FFmpeg DLLs registradas: {_ffmpeg_bin}")
    else:
        print("[AVISO] FFmpeg não encontrado no PATH. O TTS pode falhar.")

import numpy as np
import sounddevice as sd
import torch

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QFileDialog,
    QGroupBox, QProgressBar, QFrame, QLineEdit, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QIcon, QTextCursor

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
BLOCK_SIZE = 512
MAX_SILENCE_FRAMES = 15
VAD_THRESHOLD = 0.4
TTS_SAMPLE_RATE = 24000

# ─────────────────────────────────────────────────────────────
# Idiomas suportados (intersecção XTTS v2 + Whisper)
# Formato: código_xtts → (nome_exibição, código_whisper, código_marian)
# ─────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "pt":    ("Português",  "pt", "pt"),
    "en":    ("English",    "en", "en"),
    "es":    ("Español",    "es", "es"),
    "fr":    ("Français",   "fr", "fr"),
    "de":    ("Deutsch",    "de", "de"),
    "it":    ("Italiano",   "it", "it"),
    "pl":    ("Polski",     "pl", "pl"),
    "tr":    ("Türkçe",     "tr", "tr"),
    "ru":    ("Русский",    "ru", "ru"),
    "nl":    ("Nederlands", "nl", "nl"),
    "cs":    ("Čeština",    "cs", "cs"),
    "ar":    ("العربية",    "ar", "ar"),
    "zh-cn": ("中文",       "zh", "zh"),
    "ja":    ("日本語",     "ja", "ja"),
    "hu":    ("Magyar",     "hu", "hu"),
    "ko":    ("한국어",     "ko", "ko"),
    "hi":    ("हिन्दी",    "hi", "hi"),
}

# Grupo de idiomas românicos reconhecidos pelo modelo ROMANCE
_ROMANCE_LANGS = {"pt", "es", "fr", "it"}

# ─────────────────────────────────────────────────────────────
# Mapeamento de modelos MarianMT — resolução automática
# ─────────────────────────────────────────────────────────────
# Modelos multilinguais de grupo que cobrem vários pares:
#   Helsinki-NLP/opus-mt-ROMANCE-en  (pt/es/fr/it → en)
#   Helsinki-NLP/opus-mt-en-ROMANCE  (en → pt/es/fr/it)
# Para os demais, usa-se opus-mt-{src}-{tgt} ou pivot via inglês.

def _marian_code(lang_key: str) -> str:
    """Retorna o código MarianMT a partir da chave do idioma."""
    return SUPPORTED_LANGUAGES[lang_key][2]

def _is_group_model(model_name: str) -> bool:
    """Verifica se o modelo é multilingual (nome contém grupo em MAIÚSCULAS como target)."""
    # Ex: opus-mt-en-ROMANCE → target é ROMANCE (grupo)
    parts = model_name.split("/")[-1].split("-")
    # O último elemento é o target
    if parts:
        target_part = parts[-1]
        return target_part == target_part.upper() and target_part.isalpha()
    return False


def get_marian_model_names(src: str, tgt: str) -> list:
    """
    Retorna lista de modelos MarianMT a carregar para traduzir src→tgt.
    - Se src == tgt: retorna [] (sem tradução).
    - Tenta modelo direto, modelo de grupo, ou pivot via EN.
    Retorna lista de tuplas (model_name, src_code, tgt_code, needs_prefix).
    needs_prefix=True indica que o texto deve ter prefixo >>tgt<< para modelos multilinguais.
    """
    if src == tgt:
        return []

    src_m = _marian_code(src)
    tgt_m = _marian_code(tgt)

    # Caso 1: Língua românica → EN (usa modelo de grupo)
    if src in _ROMANCE_LANGS and tgt == "en":
        return [("Helsinki-NLP/opus-mt-ROMANCE-en", src_m, tgt_m, False)]

    # Caso 2: EN → Língua românica (usa modelo de grupo)
    if src == "en" and tgt in _ROMANCE_LANGS:
        return [("Helsinki-NLP/opus-mt-en-ROMANCE", src_m, tgt_m, True)]

    # Caso 3: Pivot via EN (src→en + en→tgt) — para pares não-EN
    if src != "en" and tgt != "en":
        pivot_models = []
        # src → en
        if src in _ROMANCE_LANGS:
            pivot_models.append(("Helsinki-NLP/opus-mt-ROMANCE-en", src_m, "en", False))
        else:
            pivot_models.append((f"Helsinki-NLP/opus-mt-{src_m}-en", src_m, "en", False))
        # en → tgt
        if tgt in _ROMANCE_LANGS:
            pivot_models.append(("Helsinki-NLP/opus-mt-en-ROMANCE", "en", tgt_m, True))
        else:
            pivot_models.append((f"Helsinki-NLP/opus-mt-en-{tgt_m}", "en", tgt_m, False))
        return pivot_models

    # Caso 4: Modelo direto (um dos dois é EN)
    direct = f"Helsinki-NLP/opus-mt-{src_m}-{tgt_m}"
    return [(direct, src_m, tgt_m, False)]


def resource_path(relative_path):
    """Resolve caminho de recurso para dev e exe (PyInstaller)."""
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)

DEFAULT_VOICE_FILE = resource_path("minha_voz.wav")

# ─────────────────────────────────────────────────────────────
# Worker: Carregamento de modelos (thread separada)
# ─────────────────────────────────────────────────────────────
class ModelLoaderWorker(QThread):
    """Carrega todos os modelos na GPU sem travar a GUI."""
    progress = pyqtSignal(int, str)       # (porcentagem, mensagem)
    finished_ok = pyqtSignal(object)      # dict com os modelos carregados
    finished_err = pyqtSignal(str)        # mensagem de erro

    def __init__(self, src_lang="pt", tgt_lang="en"):
        super().__init__()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def run(self):
        try:
            models = {}

            self.progress.emit(10, "Carregando VAD (Silero)...")
            vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
            )
            models["vad"] = vad_model

            self.progress.emit(30, "Carregando Whisper large-v3-turbo (CUDA)...")
            from faster_whisper import WhisperModel
            models["stt"] = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

            # --- MarianMT --- 
            mt_models_info = get_marian_model_names(self.src_lang, self.tgt_lang)
            models["mt_chain"] = []  # Lista de (tokenizer, model, src, tgt)

            if mt_models_info:
                from transformers import MarianMTModel, MarianTokenizer
                total_mt = len(mt_models_info)
                for i, (model_name, src_c, tgt_c, needs_prefix) in enumerate(mt_models_info):
                    pct = 50 + int(20 * (i / total_mt))
                    src_display = SUPPORTED_LANGUAGES.get(self.src_lang, (self.src_lang,))[0]
                    tgt_display = SUPPORTED_LANGUAGES.get(self.tgt_lang, (self.tgt_lang,))[0]
                    self.progress.emit(pct, f"Carregando MarianMT {src_display}→{tgt_display} ({i+1}/{total_mt})...")
                    tok = MarianTokenizer.from_pretrained(model_name)
                    mdl = MarianMTModel.from_pretrained(model_name).to("cuda")
                    models["mt_chain"].append((tok, mdl, src_c, tgt_c, needs_prefix))
            else:
                self.progress.emit(55, "Mesmo idioma — tradução desativada.")

            self.progress.emit(75, "Carregando Coqui XTTS v2 (CUDA) — pode demorar na 1ª vez...")
            from TTS.api import TTS
            models["tts"] = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

            self.progress.emit(100, "Todos os modelos carregados!")
            self.finished_ok.emit(models)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_err.emit(str(e))


# ─────────────────────────────────────────────────────────────
# Worker: Recarregamento de modelo MT (thread separada)
# ─────────────────────────────────────────────────────────────
class MTReloadWorker(QThread):
    """Recarrega apenas o modelo MarianMT para um novo par de idiomas."""
    progress = pyqtSignal(int, str)
    finished_ok = pyqtSignal(object)   # lista de (tokenizer, model, src, tgt)
    finished_err = pyqtSignal(str)

    def __init__(self, src_lang: str, tgt_lang: str):
        super().__init__()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def run(self):
        try:
            mt_models_info = get_marian_model_names(self.src_lang, self.tgt_lang)
            mt_chain = []

            if mt_models_info:
                from transformers import MarianMTModel, MarianTokenizer
                total = len(mt_models_info)
                for i, (model_name, src_c, tgt_c, needs_prefix) in enumerate(mt_models_info):
                    pct = int(100 * (i + 1) / total)
                    src_display = SUPPORTED_LANGUAGES.get(self.src_lang, (self.src_lang,))[0]
                    tgt_display = SUPPORTED_LANGUAGES.get(self.tgt_lang, (self.tgt_lang,))[0]
                    self.progress.emit(pct, f"Recarregando MT {src_display}→{tgt_display} ({i+1}/{total})...")
                    tok = MarianTokenizer.from_pretrained(model_name)
                    mdl = MarianMTModel.from_pretrained(model_name).to("cuda")
                    mt_chain.append((tok, mdl, src_c, tgt_c, needs_prefix))
            else:
                self.progress.emit(100, "Mesmo idioma — tradução desativada.")

            self.finished_ok.emit(mt_chain)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_err.emit(str(e))


# ─────────────────────────────────────────────────────────────
# Worker: Pipeline de tradução (thread separada)
# ─────────────────────────────────────────────────────────────
class TranslationWorker(QThread):
    """Consome a fila de áudio e executa STT → MT → TTS."""
    new_transcription = pyqtSignal(str, str, str, str)  # (src_code, texto_src, tgt_code, texto_tgt)
    status_update = pyqtSignal(str)                     # status textual

    def __init__(self, models: dict, audio_queue: queue.Queue,
                 voice_file: str, output_device: int,
                 src_lang: str, tgt_lang: str):
        super().__init__()
        self.models = models
        self.audio_queue = audio_queue
        self.voice_file = voice_file
        self.output_device = output_device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        stt = self.models["stt"]
        mt_chain = self.models["mt_chain"]  # Lista de (tok, model, src_c, tgt_c, needs_prefix)
        tts = self.models["tts"]

        # Código Whisper para STT
        whisper_lang = SUPPORTED_LANGUAGES[self.src_lang][1]
        # Código XTTS para TTS
        xtts_lang = self.tgt_lang

        while self._running:
            try:
                audio_data = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # --- STT ---
            self.status_update.emit("⚙️ Transcrevendo...")
            segments, _ = stt.transcribe(
                audio_data, beam_size=5, language=whisper_lang,
            )
            src_text = "".join(seg.text for seg in segments).strip()
            if not src_text:
                self.status_update.emit("🔴 Silêncio (sem texto)")
                continue

            # --- MT ---
            if mt_chain:
                self.status_update.emit("⚙️ Traduzindo...")
                translated_text = src_text
                for tok, model, src_c, tgt_c, needs_prefix in mt_chain:
                    # Modelos multilinguais (ex: en-ROMANCE) precisam do prefixo >>tgt<<
                    prefix = ""
                    if needs_prefix and tgt_c:
                        prefix = f">>{tgt_c}<< "
                        print(f"[DEBUG] MT prefix: {prefix.strip()}")
                    
                    input_text = prefix + translated_text
                    inputs = tok(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")
                    with torch.no_grad():
                        translated_tokens = model.generate(**inputs, max_length=512)
                    translated_text = tok.decode(translated_tokens[0], skip_special_tokens=True)
                tgt_text = translated_text
            else:
                # Mesmo idioma — sem tradução
                tgt_text = src_text

            self.new_transcription.emit(
                self.src_lang, src_text, self.tgt_lang, tgt_text
            )

            # --- TTS ---
            self.status_update.emit("🔊 Sintetizando voz...")
            try:
                wav = tts.tts(text=tgt_text, speaker_wav=self.voice_file, language=xtts_lang)
                wav_np = np.array(wav, dtype=np.float32)

                # Normaliza o volume para evitar clipping
                peak = np.max(np.abs(wav_np))
                if peak > 0:
                    wav_np = wav_np / peak * 0.9

                print(f"[DEBUG] Áudio gerado: {len(wav_np)} samples, peak={peak:.4f}, device={self.output_device}")

                # Reproduz usando OutputStream explícito (evita conflito com InputStream global)
                self.status_update.emit("🔊 Reproduzindo áudio...")
                with sd.OutputStream(
                    samplerate=TTS_SAMPLE_RATE,
                    channels=1,
                    dtype='float32',
                    device=self.output_device,
                ) as out_stream:
                    out_stream.write(wav_np.reshape(-1, 1))

                print("[DEBUG] Áudio reproduzido com sucesso.")
            except Exception as e:
                print(f"[DEBUG] Erro TTS/Playback: {e}")
                self.status_update.emit(f"❌ Erro TTS: {e}")

            self.status_update.emit("🟢 Pronto — aguardando fala...")


# ─────────────────────────────────────────────────────────────
# Janela principal
# ─────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioTranslate — Tradutor de Voz em Tempo Real")
        self.setMinimumSize(750, 660)
        self.resize(820, 720)

        # Estado
        self.models = None
        self.audio_queue = queue.Queue()
        self.stream = None
        self.translation_worker = None
        self.is_capturing = False
        self.mt_reload_worker = None

        # VAD state
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_frames = 0

        # Idiomas selecionados
        self.src_lang = "pt"
        self.tgt_lang = "en"

        self._build_ui()
        self._apply_styles()
        self._start_loading_models()

    # ── UI ──────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        # --- Header ---
        header = QLabel("🎙️ AudioTranslate")
        header.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(header)

        self.subtitle = QLabel()
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setStyleSheet("color: #888; font-size: 12px;")
        self._update_subtitle()
        root_layout.addWidget(self.subtitle)

        # --- Progress bar (loading) ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Aguardando...")
        self.progress_bar.setValue(0)
        root_layout.addWidget(self.progress_bar)

        # --- Configurações ---
        config_group = QGroupBox("⚙️ Configurações")
        config_layout = QVBoxLayout(config_group)

        # --- Seleção de idiomas ---
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("🌐 Origem:"))
        self.combo_src_lang = QComboBox()
        self.combo_src_lang.setMinimumWidth(160)
        lang_row.addWidget(self.combo_src_lang, 1)

        self.btn_swap_lang = QPushButton("⇄")
        self.btn_swap_lang.setFixedSize(42, 32)
        self.btn_swap_lang.setToolTip("Trocar idiomas")
        self.btn_swap_lang.clicked.connect(self._swap_languages)
        lang_row.addWidget(self.btn_swap_lang)

        lang_row.addWidget(QLabel("🎯 Destino:"))
        self.combo_tgt_lang = QComboBox()
        self.combo_tgt_lang.setMinimumWidth(160)
        lang_row.addWidget(self.combo_tgt_lang, 1)

        config_layout.addLayout(lang_row)

        # Popula combos de idiomas
        self._lang_keys = list(SUPPORTED_LANGUAGES.keys())
        for key in self._lang_keys:
            display_name = f"{SUPPORTED_LANGUAGES[key][0]} ({key})"
            self.combo_src_lang.addItem(display_name)
            self.combo_tgt_lang.addItem(display_name)

        # Pré-seleciona PT → EN
        self.combo_src_lang.setCurrentIndex(self._lang_keys.index("pt"))
        self.combo_tgt_lang.setCurrentIndex(self._lang_keys.index("en"))

        # Conecta sinais de mudança de idioma
        self.combo_src_lang.currentIndexChanged.connect(self._on_language_changed)
        self.combo_tgt_lang.currentIndexChanged.connect(self._on_language_changed)

        # Input device
        row_in = QHBoxLayout()
        row_in.addWidget(QLabel("🎤 Entrada:"))
        self.combo_input = QComboBox()
        self.combo_input.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._populate_input_devices()
        row_in.addWidget(self.combo_input, 1)
        config_layout.addLayout(row_in)

        # Output device
        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("🔈 Saída:"))
        self.combo_output = QComboBox()
        self.combo_output.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._populate_output_devices()
        row_out.addWidget(self.combo_output, 1)
        config_layout.addLayout(row_out)

        # Voice file
        row_voice = QHBoxLayout()
        row_voice.addWidget(QLabel("🗣️ Voz referência:"))
        self.txt_voice = QLineEdit(DEFAULT_VOICE_FILE)
        self.txt_voice.setReadOnly(True)
        row_voice.addWidget(self.txt_voice, 1)
        btn_browse = QPushButton("Procurar")
        btn_browse.setFixedWidth(90)
        btn_browse.clicked.connect(self._browse_voice_file)
        row_voice.addWidget(btn_browse)
        config_layout.addLayout(row_voice)

        root_layout.addWidget(config_group)

        # --- Controles ---
        ctrl_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶  Iniciar")
        self.btn_start.setFixedHeight(42)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self._toggle_capture)
        ctrl_layout.addWidget(self.btn_start)

        self.btn_refresh = QPushButton("🔄 Atualizar dispositivos")
        self.btn_refresh.setFixedHeight(42)
        self.btn_refresh.clicked.connect(self._refresh_devices)
        ctrl_layout.addWidget(self.btn_refresh)
        root_layout.addLayout(ctrl_layout)

        # --- Log de transcrição ---
        log_group = QGroupBox("📝 Transcrição / Tradução")
        log_layout = QVBoxLayout(log_group)
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setFont(QFont("Consolas", 11))
        log_layout.addWidget(self.text_log)
        root_layout.addWidget(log_group, 1)

        # --- Status bar ---
        self.status_label = QLabel("⏳ Carregando modelos...")
        self.status_label.setStyleSheet("color: #aaa; padding: 4px;")
        root_layout.addWidget(self.status_label)

    def _update_subtitle(self):
        src_name = SUPPORTED_LANGUAGES[self.src_lang][0]
        tgt_name = SUPPORTED_LANGUAGES[self.tgt_lang][0]
        self.subtitle.setText(
            f"Tradução de voz em tempo real  •  {src_name} → {tgt_name}  •  GPU Accelerated"
        )

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QWidget {
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 8px;
                margin-top: 10px;
                padding: 14px 10px 10px 10px;
                font-weight: bold;
                font-size: 13px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
            }
            QComboBox, QLineEdit {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 6px 10px;
                color: #cdd6f4;
                font-size: 12px;
            }
            QComboBox::drop-down {
                border: none;
                width: 28px;
            }
            QComboBox QAbstractItemView {
                background-color: #313244;
                color: #cdd6f4;
                selection-background-color: #585b70;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #74c7ec;
            }
            QPushButton:pressed {
                background-color: #94e2d5;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
            QTextEdit {
                background-color: #181825;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
            }
            QProgressBar {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                text-align: center;
                color: #cdd6f4;
                font-size: 11px;
                height: 22px;
            }
            QProgressBar::chunk {
                background-color: #a6e3a1;
                border-radius: 5px;
            }
            QLabel {
                font-size: 12px;
            }
        """)

    # ── Idiomas ─────────────────────────────────────────────
    def _on_language_changed(self):
        """Chamado quando o usuário muda o idioma de origem ou destino."""
        new_src = self._lang_keys[self.combo_src_lang.currentIndex()]
        new_tgt = self._lang_keys[self.combo_tgt_lang.currentIndex()]

        src_changed = new_src != self.src_lang
        tgt_changed = new_tgt != self.tgt_lang

        self.src_lang = new_src
        self.tgt_lang = new_tgt
        self._update_subtitle()

        # Se modelos já estão carregados e a captura NÃO está ativa,
        # recarrega o modelo MT para o novo par
        if self.models is not None and not self.is_capturing and (src_changed or tgt_changed):
            self._reload_mt_model()

    def _swap_languages(self):
        """Troca idiomas de origem e destino."""
        src_idx = self.combo_src_lang.currentIndex()
        tgt_idx = self.combo_tgt_lang.currentIndex()

        # Bloquear sinais para evitar recarregamento duplo
        self.combo_src_lang.blockSignals(True)
        self.combo_tgt_lang.blockSignals(True)

        self.combo_src_lang.setCurrentIndex(tgt_idx)
        self.combo_tgt_lang.setCurrentIndex(src_idx)

        self.combo_src_lang.blockSignals(False)
        self.combo_tgt_lang.blockSignals(False)

        # Dispara manualmente o handler
        self._on_language_changed()

    def _reload_mt_model(self):
        """Recarrega apenas o modelo MarianMT para o par selecionado."""
        if self.mt_reload_worker and self.mt_reload_worker.isRunning():
            return  # Já recarregando

        self.btn_start.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Recarregando modelo MT...")
        self.status_label.setText("⏳ Recarregando modelo de tradução...")

        self.mt_reload_worker = MTReloadWorker(self.src_lang, self.tgt_lang)
        self.mt_reload_worker.progress.connect(self._on_load_progress)
        self.mt_reload_worker.finished_ok.connect(self._on_mt_reloaded)
        self.mt_reload_worker.finished_err.connect(self._on_mt_reload_error)
        self.mt_reload_worker.start()

    def _on_mt_reloaded(self, mt_chain):
        """Chamado quando o modelo MT foi recarregado com sucesso."""
        # Libera modelos antigos
        if self.models and "mt_chain" in self.models:
            for tok, mdl, *_ in self.models["mt_chain"]:
                del mdl
                del tok
            torch.cuda.empty_cache()

        self.models["mt_chain"] = mt_chain
        self.btn_start.setEnabled(True)

        src_name = SUPPORTED_LANGUAGES[self.src_lang][0]
        tgt_name = SUPPORTED_LANGUAGES[self.tgt_lang][0]

        if mt_chain:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat(f"✅ MT {src_name}→{tgt_name} pronto!")
            self.status_label.setText(f"✅ Modelo MT recarregado: {src_name} → {tgt_name}")
            self._log_info(f"Modelo de tradução atualizado: {src_name} → {tgt_name}")
        else:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat(f"✅ Mesmo idioma — tradução desativada")
            self.status_label.setText(f"✅ Mesmo idioma selecionado — tradução desativada.")
            self._log_info(f"Mesmo idioma ({src_name}) — áudio será apenas transcrito.")

    def _on_mt_reload_error(self, err):
        """Chamado quando o recarregamento do modelo MT falhou."""
        self.btn_start.setEnabled(True)
        self.progress_bar.setFormat(f"❌ Erro MT: {err}")
        self.status_label.setText(f"❌ Falha ao recarregar MT: {err}")
        self._log_error(f"Erro ao recarregar modelo de tradução:\n{err}")

    # ── Dispositivos ────────────────────────────────────────
    def _safe_device_name(self, dev):
        name = dev["name"]
        try:
            name = name.encode("latin-1").decode("utf-8")
        except Exception:
            pass
        return name

    def _populate_input_devices(self):
        self.combo_input.clear()
        self._input_device_ids = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                name = self._safe_device_name(dev)
                self.combo_input.addItem(f"[{i}] {name}")
                self._input_device_ids.append(i)

    def _populate_output_devices(self):
        self.combo_output.clear()
        self._output_device_ids = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] > 0:
                name = self._safe_device_name(dev)
                self.combo_output.addItem(f"[{i}] {name}")
                self._output_device_ids.append(i)

    def _refresh_devices(self):
        sd._terminate()
        sd._initialize()
        self._populate_input_devices()
        self._populate_output_devices()
        self._log_info("Dispositivos atualizados.")

    def _browse_voice_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar arquivo de voz",
            os.path.dirname(DEFAULT_VOICE_FILE),
            "Audio Files (*.wav *.mp3 *.m4a *.flac);;All Files (*)"
        )
        if path:
            self.txt_voice.setText(path)

    # ── Carregamento de modelos ─────────────────────────────
    def _start_loading_models(self):
        self.loader = ModelLoaderWorker(self.src_lang, self.tgt_lang)
        self.loader.progress.connect(self._on_load_progress)
        self.loader.finished_ok.connect(self._on_models_loaded)
        self.loader.finished_err.connect(self._on_models_error)
        self.loader.start()

    def _on_load_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(msg)
        self.status_label.setText(f"⏳ {msg}")

    def _on_models_loaded(self, models):
        self.models = models
        self.btn_start.setEnabled(True)
        self.progress_bar.setFormat("✅ Modelos prontos!")
        self.status_label.setText("✅ Pronto — selecione os dispositivos e clique em Iniciar.")
        self._log_info("Todos os modelos carregados na GPU com sucesso.")

    def _on_models_error(self, err):
        self.progress_bar.setFormat(f"❌ Erro: {err}")
        self.status_label.setText(f"❌ Falha ao carregar modelos: {err}")
        self._log_error(f"Erro ao carregar modelos:\n{err}")

    # ── Captura de áudio ───────────────────────────────────
    def _toggle_capture(self):
        if self.is_capturing:
            self._stop_capture()
        else:
            self._start_capture()

    def _start_capture(self):
        if self.combo_input.count() == 0 or self.combo_output.count() == 0:
            self._log_error("Nenhum dispositivo de entrada ou saída encontrado.")
            return

        voice_file = self.txt_voice.text()
        if not os.path.isfile(voice_file):
            self._log_error(f"Arquivo de voz não encontrado: {voice_file}")
            return

        input_idx = self.combo_input.currentIndex()
        self.selected_input = self._input_device_ids[input_idx]

        output_idx = self.combo_output.currentIndex()
        self.selected_output = self._output_device_ids[output_idx]

        # Reset VAD state
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_frames = 0

        # Drain any old data
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()

        # Start translation worker
        self.translation_worker = TranslationWorker(
            self.models, self.audio_queue, voice_file, self.selected_output,
            self.src_lang, self.tgt_lang
        )
        self.translation_worker.new_transcription.connect(self._on_transcription)
        self.translation_worker.status_update.connect(self._on_worker_status)
        self.translation_worker.start()

        # Start audio stream
        try:
            self.stream = sd.InputStream(
                device=self.selected_input,
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=BLOCK_SIZE,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as e:
            self._log_error(f"Erro ao abrir microfone: {e}")
            self.translation_worker.stop()
            return

        self.is_capturing = True
        self.btn_start.setText("⏹  Parar")
        self.btn_start.setStyleSheet(
            "background-color: #f38ba8; color: #1e1e2e;"
        )
        self.combo_input.setEnabled(False)
        self.combo_output.setEnabled(False)
        self.combo_src_lang.setEnabled(False)
        self.combo_tgt_lang.setEnabled(False)
        self.btn_swap_lang.setEnabled(False)

        src_name = SUPPORTED_LANGUAGES[self.src_lang][0]
        tgt_name = SUPPORTED_LANGUAGES[self.tgt_lang][0]
        self.status_label.setText(f"🟢 Capturando ({src_name} → {tgt_name}) — fale no microfone.")
        self._log_info(f"Captura iniciada ({src_name} → {tgt_name}). Fale no microfone...")

    def _stop_capture(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.translation_worker:
            self.translation_worker.stop()
            self.translation_worker.wait(3000)
            self.translation_worker = None

        self.is_capturing = False
        self.btn_start.setText("▶  Iniciar")
        self.btn_start.setStyleSheet("")
        self.combo_input.setEnabled(True)
        self.combo_output.setEnabled(True)
        self.combo_src_lang.setEnabled(True)
        self.combo_tgt_lang.setEnabled(True)
        self.btn_swap_lang.setEnabled(True)
        self.status_label.setText("⏸️ Parado.")
        self._log_info("Captura encerrada.")

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback do sounddevice — roda numa thread de áudio."""
        chunk = indata[:, 0].astype(np.float32)
        audio_tensor = torch.from_numpy(chunk.copy())

        with torch.no_grad():
            speech_prob = self.models["vad"](audio_tensor, SAMPLE_RATE).item()

        if speech_prob > VAD_THRESHOLD:
            self.is_speaking = True
            self.silence_frames = 0
            self.audio_buffer.append(chunk)
        else:
            if self.is_speaking:
                self.silence_frames += 1
                self.audio_buffer.append(chunk)
                if self.silence_frames > MAX_SILENCE_FRAMES:
                    self.is_speaking = False
                    full_audio = np.concatenate(self.audio_buffer)
                    self.audio_queue.put(full_audio)
                    self.audio_buffer = []

    # ── Slots de atualização ────────────────────────────────
    def _on_transcription(self, src_code, src_text, tgt_code, tgt_text):
        src_name = SUPPORTED_LANGUAGES[src_code][0]
        tgt_name = SUPPORTED_LANGUAGES[tgt_code][0]
        src_upper = src_code.upper()
        tgt_upper = tgt_code.upper()

        self.text_log.append(
            f'<span style="color:#89b4fa;font-weight:bold;">[{src_upper}]</span> {src_text}'
        )
        if src_code != tgt_code:
            self.text_log.append(
                f'<span style="color:#a6e3a1;font-weight:bold;">[{tgt_upper}]</span> {tgt_text}'
            )
        self.text_log.append("")
        # Auto-scroll
        cursor = self.text_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_log.setTextCursor(cursor)

    def _on_worker_status(self, msg):
        self.status_label.setText(msg)



    # ── Logging helpers ─────────────────────────────────────
    def _log_info(self, msg):
        self.text_log.append(
            f'<span style="color:#94e2d5;">ℹ️ {msg}</span>'
        )

    def _log_error(self, msg):
        self.text_log.append(
            f'<span style="color:#f38ba8;">❌ {msg}</span>'
        )

    # ── Cleanup ─────────────────────────────────────────────
    def closeEvent(self, event):
        self._stop_capture()
        event.accept()


# ─────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
