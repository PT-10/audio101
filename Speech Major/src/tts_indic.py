from transformers import AutoModel
import numpy as np
import soundfile as sf

# Load IndicF5 from Hugging Face
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Generate speech
audio = model(
    "ఏంటి ప్రాంథీ, ఇది అంతా కూడా మీరు అర్థం గా లేదు?",
    ref_audio_path="src/outputs/output.wav",
    ref_text="Hi, this is me, Prashant. I am generating a six second clip for voice cloning. The quick brown fox jumped over the lazy dogs."
)

# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
sf.write("src/outputs/namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
print("Audio saved succesfully.")
