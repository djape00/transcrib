from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import tempfile
import os
import time

app = Flask(**name**)

print("Loading Whisper model...")

model = WhisperModel(
"small",
device="cpu",
compute_type="int8"
)

print("Model loaded!")

@app.route('/transcribe', methods=['POST'])
def transcribe():
try:
if 'file' not in request.files:
return jsonify({"error": "No file provided"}), 400

```
    file = request.files['file']

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        temp_path = tmp.name

    try:
        start = time.time()

        segments, info = model.transcribe(
            temp_path,
            language="sr",
            beam_size=5,
            vad_filter=True
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        elapsed = time.time() - start

        print(f"Detected language: {info.language}")
        print(f"Language probability: {info.language_probability}")
        print(f"Finished in {elapsed:.2f}s")

        os.unlink(temp_path)

        return jsonify({
            "text": text,
            "language": info.language,
            "time_seconds": round(elapsed, 2)
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

        return jsonify({
            "error": str(e)
        }), 500

except Exception as e:
    return jsonify({
        "error": str(e)
    }), 500
```

@app.route('/health', methods=['GET'])
def health():
return jsonify({
"status": "ok"
})

if **name** == '**main**':
app.run(
host='0.0.0.0',
port=5000
)

