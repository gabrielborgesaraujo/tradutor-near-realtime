import sounddevice as sd
import numpy as np
import torch
import warnings
import ssl
import queue
from faster_whisper import WhisperModel

ssl._create_default_https_context = ssl._create_unverified_context 
warnings.filterwarnings("ignore")

# Configurações de Áudio
SAMPLE_RATE = 16000 
BLOCK_SIZE = 512    

print("Carregando modelos na GPU (isso pode levar alguns segundos na primeira vez)...")

# 1. Carrega o VAD
vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              trust_repo=True)

# 2. Carrega o Faster-Whisper (Modelo Small é excelente para near real-time)
stt_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

# Variáveis de Controle de Estado
audio_queue = queue.Queue()
audio_buffer = []
is_speaking = False
silence_frames = 0
MAX_SILENCE_FRAMES = 15 # Aprox. 0.5 segundos de silêncio para considerar fim da frase

def audio_callback(indata, frames, time, status):
    global is_speaking, audio_buffer, silence_frames

    # Converte para float32 (formato exigido pelo Whisper)
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
            
            # Se ficou em silêncio tempo suficiente, corta a frase e envia para processar
            if silence_frames > MAX_SILENCE_FRAMES:
                is_speaking = False
                # Junta todos os blocos de áudio em um único array
                full_audio = np.concatenate(audio_buffer)
                audio_queue.put(full_audio)
                audio_buffer = [] # Limpa o buffer para a próxima frase
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
    print(f"\nIniciando... Fale pausadamente para testar. (Ctrl+C para sair)\n")

    # Inicia a thread de gravação
    with sd.InputStream(device=device_id, samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback):
        # Loop principal (Thread de Processamento)
        while True:
            # Verifica se há áudio na fila para transcrever
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                
                # Transcreve usando o Whisper
                segments, info = stt_model.transcribe(audio_data, beam_size=5, language="pt")
                
                text = "".join([segment.text for segment in segments]).strip()
                
                # Só imprime se realmente captou alguma palavra
                if text:
                    print(f"\n[Você]: {text}\n")
                    
            sd.sleep(50) # Pausa curta para não sobrecarregar a CPU no loop infinito

except KeyboardInterrupt:
    print("\n\nEncerrado.")