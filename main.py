from flask import Flask, request, jsonify
import whisper
import tempfile
import os

app = Flask(__name__)

print("Loading Whisper model...")
model = whisper.load_model("tiny")
print("Model loaded!")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            
            try:
                result = model.transcribe(tmp.name, language="sr")
                os.unlink(tmp.name)
                
                return jsonify({"text": result["text"]})
            except Exception as e:
                os.unlink(tmp.name)
                return jsonify({"error": str(e)}), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
