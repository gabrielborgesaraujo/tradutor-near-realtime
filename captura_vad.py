import sounddevice as sd
import numpy as np
import torch
import warnings
import ssl

# Ignora a verificação de certificado SSL para permitir o download
ssl._create_default_https_context = ssl._create_unverified_context 
warnings.filterwarnings("ignore")

# Configurações de Áudio
SAMPLE_RATE = 16000 
BLOCK_SIZE = 512    

print("Carregando modelo Silero VAD na memória...")
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              trust_repo=True)

def audio_callback(indata, frames, time, status):
    if status:
        pass # Ignora pequenos avisos de status para não sujar o console

    audio_tensor = torch.from_numpy(indata[:, 0].copy())

    with torch.no_grad(): 
        speech_prob = model(audio_tensor, SAMPLE_RATE).item()

    # Threshold de 50% de confiança
    if speech_prob > 0.5:
        print(f"🟢 Falando...   (Prob: {speech_prob:.2f})    ", end='\r')
    else:
        print(f"🔴 Silêncio... (Prob: {speech_prob:.2f})    ", end='\r')


# --- NOVA PARTE: SELEÇÃO DE DISPOSITIVO ---
print("\n--- Dispositivos de Entrada Disponíveis ---")
devices = sd.query_devices()
input_devices = []

for i, dev in enumerate(devices):
    # Filtra apenas dispositivos que possuem canais de entrada (microfones/loopback)
    if dev['max_input_channels'] > 0:
        # Pega o nome do dispositivo e garante que os caracteres apareçam corretamente
        name = dev['name']
        try:
            name = name.encode('latin-1').decode('utf-8')
        except:
            pass
        print(f"[{i}] {name}")
        input_devices.append(i)

print("-" * 43)

try:
    # Solicita ao usuário que escolha o ID do dispositivo
    device_id = int(input("Digite o número [ID] do microfone que deseja usar: "))
    
    print(f"\nIniciando captura no dispositivo [{device_id}]...")
    print("Fale alguma coisa para testar. Pressione Ctrl+C para sair.\n")

    # Passa o argumento 'device' com o ID escolhido
    with sd.InputStream(device=device_id, samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback):
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\n\nCaptura encerrada pelo usuário.")
except ValueError:
    print("\n\nErro: Você precisa digitar um número válido.")
except Exception as e:
    print(f"\n\nErro ao acessar o microfone: {e}")