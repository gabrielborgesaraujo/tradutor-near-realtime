import sounddevice as sd
import numpy as np
import torch
import warnings
import ssl
import queue
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import MarianMTModel, MarianTokenizer

ssl._create_default_https_context = ssl._create_unverified_context 
warnings.filterwarnings("ignore")

# Configurações de Áudio
SAMPLE_RATE = 16000 
BLOCK_SIZE = 512    

print("Carregando modelos na GPU (aproveitando os 12GB de VRAM)...")

# 1. Carrega o VAD
vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              trust_repo=True)

# 2. Carrega o Faster-Whisper
stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

# 3. Carrega o modelo especialista PT -> EN (Helsinki-NLP)
print("Carregando modelo de tradução ultra-rápido (Helsinki PT-EN)...")
model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translator_model = MarianMTModel.from_pretrained(model_name).to("cuda")

# Variáveis de Controle de Estado
audio_queue = queue.Queue()
audio_buffer = []
is_speaking = False
silence_frames = 0
MAX_SILENCE_FRAMES = 30 # Aprox. 1 segundo de silêncio para fechar a frase

def translate_to_english(text):
    """
    Traduz de PT para EN usando o modelo otimizado MarianMT na GPU.
    """
    # Prepara o texto
    inputs = tokenizer(text, return_tensors="pt", padding=True).to("cuda")
    
    # Gera a tradução instantaneamente
    translated_tokens = translator_model.generate(**inputs, max_length=100)
    
    # Decodifica e retorna
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

def audio_callback(indata, frames, time, status):
    global is_speaking, audio_buffer, silence_frames

    chunk = indata[:, 0].astype(np.float32)
    audio_tensor = torch.from_numpy(chunk.copy())

    with torch.no_grad(): 
        speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

    # Threshold de 0.4 para captar melhor o início das palavras
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

# --- SELEÇÃO DE DISPOSITIVO ---
print("\n--- Dispositivos de Entrada ---")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        name = dev['name']
        try: name = name.encode('latin-1').decode('utf-8')
        except: pass
        print(f"[{i}] {name}")

try:
    device_id = int(input("\nDigite o ID do microfone: "))
    print(f"\nIniciando... Fale para testar. (Ctrl+C para sair)\n")

    with sd.InputStream(device=device_id, samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback):
        while True:
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                
                # 1. Transcreve (PT)
                segments, info = stt_model.transcribe(
                    audio_data, 
                    beam_size=5, 
                    language="pt",
                    initial_prompt="Olá, tudo bem? Esta é uma transcrição clara e com pontuação."
                )
                pt_text = "".join([segment.text for segment in segments]).strip()
                
                if pt_text:
                    # 2. Traduz (EN)
                    en_text = translate_to_english(pt_text)
                    
                    # 3. Exibe o resultado lado a lado
                    print(f"\n[PT]: {pt_text}")
                    print(f"[EN]: {en_text}\n")
                    
            sd.sleep(50)

except KeyboardInterrupt:
    print("\n\nEncerrado.")