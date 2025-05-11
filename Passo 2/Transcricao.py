import whisper
import torch
import numpy as np
from datetime import timedelta

# Limpar VRAM (se estiver usando GPU)
torch.cuda.empty_cache()

# Carregar o modelo no CPU
model = whisper.load_model("large-v2", device="cpu")

# Caminho do áudio
audio_path = "audio.wav"

# Parâmetros do corte
start_seconds = 1080
end_seconds = 1740 

# Carregar o áudio completo
audio = whisper.load_audio(audio_path)

# Cortar entre os tempos desejados
sample_rate = whisper.audio.SAMPLE_RATE
# start_sample = int(start_seconds * sample_rate)
# end_sample = int(end_seconds * sample_rate)

# (Opcional) Inserir 2 segundos de silêncio no início
silence = np.zeros(int(2 * sample_rate))
audio = np.concatenate((silence, audio)).astype(np.float32)

# Transcrição
result = model.transcribe(
    audio=audio,
    language="pt",
    fp16=False,
    temperature=0.5,
    best_of=3,
    beam_size=3,
    verbose=True,
    condition_on_previous_text=False
)

# Salvar resultado com tempos em formato HH:MM:SS
with open("transcricao_com_tempos.txt", "w", encoding="utf-8") as f:
    for segment in result["segments"]:
        start_td = timedelta(seconds=round(segment["start"]))
        end_td = timedelta(seconds=round(segment["end"]))
        text = segment["text"].strip()
        f.write(f"[{str(start_td)} - {str(end_td)}] {text}\n")
