import sounddevice as sd
import numpy as np
import torch
import warnings
import ssl
import queue
import threading
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS

# Ignora avisos para manter o console limpo
ssl._create_default_https_context = ssl._create_unverified_context 
warnings.filterwarnings("ignore")

# --- 1. CARREGAMENTO DOS MODELOS (GPU) ---
print("Iniciando a ignição dos modelos na RTX 4070 Super...")

# VAD (Silero)
print("[1/4] Carregando VAD...")
vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)

# STT (Faster-Whisper Medium)
print("[2/4] Carregando Whisper...")
stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

# MT (Helsinki PT -> EN)
print("[3/4] Carregando Tradutor...")
mt_name = "Helsinki-NLP/opus-mt-ROMANCE-en"
mt_tokenizer = MarianTokenizer.from_pretrained(mt_name)
mt_model = MarianMTModel.from_pretrained(mt_name).to("cuda")

# TTS (Coqui XTTSv2)
print("[4/4] Carregando Coqui XTTSv2 (Isso pode levar um tempinho na 1ª vez)...")
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
ARQUIVO_VOZ = "C:\\Projetos\\AudioTranslate\\minha_voz.wav" # <-- O SEU MOLDE DE VOZ

print("\n✅ Todos os modelos carregados com sucesso!\n")

# --- 2. CONFIGURAÇÕES E ESTADOS ---
SAMPLE_RATE = 16000 
BLOCK_SIZE = 512    
MAX_SILENCE_FRAMES = 15 

audio_queue = queue.Queue()
audio_buffer = []
is_speaking = False
silence_frames = 0

# --- 3. FUNÇÕES PRINCIPAIS ---
def translate_to_english(text):
    inputs = mt_tokenizer(text, return_tensors="pt", padding=True).to("cuda")
    translated_tokens = mt_model.generate(**inputs, max_length=100)
    return mt_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

def process_pipeline():
    """Thread dedicada para processar a fila de áudios sem travar o microfone"""
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            
            # 1. Transcreve
            segments, _ = stt_model.transcribe(audio_data, beam_size=5, language="pt", initial_prompt="Olá, tudo bem? Transcrição clara.")
            pt_text = "".join([segment.text for segment in segments]).strip()
            
            if pt_text:
                print(f"\n[PT]: {pt_text}")
                
                # 2. Traduz
                en_text = translate_to_english(pt_text)
                print(f"[EN]: {en_text}")
                
                # 3. Sintetiza a voz e toca
                print("🔊 Gerando áudio...", end="\r")
                try:
                    # O XTTSv2 gera o áudio em 24000Hz por padrão
                    wav = tts_model.tts(text=en_text, speaker_wav=ARQUIVO_VOZ, language="en")
                    sd.play(wav, samplerate=24000)
                    sd.wait() # Aguarda terminar de falar antes de liberar o console
                    print("✅ Áudio reproduzido.  ", end="\r")
                except Exception as e:
                    print(f"\nErro no TTS: Certifique-se de que '{ARQUIVO_VOZ}' está na pasta. Detalhe: {e}")
                    
def audio_callback(indata, frames, time, status):
    global is_speaking, audio_buffer, silence_frames

    chunk = indata[:, 0].astype(np.float32)
    audio_tensor = torch.from_numpy(chunk.copy())

    with torch.no_grad(): 
        speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

    if speech_prob > 0.4:
        is_speaking = True
        silence_frames = 0
        audio_buffer.append(chunk)
        print("🟢 Falando...                ", end='\r')
    else:
        if is_speaking:
            silence_frames += 1
            audio_buffer.append(chunk)
            
            if silence_frames > MAX_SILENCE_FRAMES:
                is_speaking = False
                full_audio = np.concatenate(audio_buffer)
                audio_queue.put(full_audio)
                audio_buffer = [] 
                print("⚙️ Processando...             ", end='\r')
        else:
            print("🔴 Silêncio...                ", end='\r')

# --- 4. EXECUÇÃO ---
print("--- Dispositivos de Entrada ---")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        name = dev['name']
        try: name = name.encode('latin-1').decode('utf-8')
        except: pass
        print(f"[{i}] {name}")

try:
    device_id = int(input("\nDigite o ID do microfone: "))
    
    # Inicia a Thread de Processamento da IA em paralelo
    threading.Thread(target=process_pipeline, daemon=True).start()
    
    print(f"\nIniciando sistema... Fale no microfone. (Ctrl+C para sair)\n")

    # Inicia a gravação contínua do microfone
    with sd.InputStream(device=device_id, samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback):
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\n\nEncerrado.")