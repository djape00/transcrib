from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import tempfile
import os
import time
import subprocess
import shutil

app = Flask(__name__)

print("Loading Whisper LARGE-V3 model...")
model = WhisperModel(
    "medium",  # LARGE MODEL!
    device="cpu",
    compute_type="int8"  # Štedi memoriju
)
print("Model loaded!")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        try:
            start = time.time()
            
            # =========================
            # 1. AUDIO CLEANUP (OPTIMIZOVANO)
            # =========================
            clean_audio = temp_path + "_clean.wav"
            subprocess.run([
                "ffmpeg",
                "-y",
                "-i", temp_path,
                "-ar", "16000",
                "-ac", "1",
                "-af", "highpass=f=80,lowpass=f=8000,afftdn=nf=15,adeclick=t=0.1",
                clean_audio
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # =========================
            # 2. WHISPER TRANSCRIPTION - OPTIMIZOVANO
            # =========================
            segments, info = model.transcribe(
                clean_audio,
                language="sr",                    # SRPSKI!
                beam_size=10,                     # VEĆI BEAM
                best_of=5,                        # VIŠE OPCIJA
                vad_filter=True,                  # PRESKAKANJE TIŠINE
                vad_parameters={
                    "threshold": 0.4,
                    "min_duration": 0.5
                },
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # TESTIRA VIŠE
                repetition_penalty=1.2            # SPRJEČAVA PONAVLJANJE
            )
            
            text = " ".join(
                segment.text.strip()
                for segment in segments
            )
            
            elapsed = time.time() - start
            
            print(f"Detected language: {info.language}")
            print(f"Language probability: {info.language_probability}")
            print(f"Finished in {elapsed:.2f}s")
            
            # Cleanup
            os.unlink(temp_path)
            os.unlink(clean_audio)
            
            return jsonify({
                "text": text,
                "language": info.language,
                "language_probability": info.language_probability,
                "time_seconds": round(elapsed, 2)
            })
        
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return jsonify({"error": str(e)}), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
