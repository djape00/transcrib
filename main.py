from flask import Flask, request, jsonify
import whisper
import os
import tempfile

app = Flask(__name__)

# Preload model
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded!")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        file.save(tmp.name)
        result = model.transcribe(tmp.name, language="sr")
        os.unlink(tmp.name)
    
    return jsonify({"text": result["text"]})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
