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
    print("=== ZAHTJEV PRIMLJEN ===")
    print(f"Request files: {request.files}")
    print(f"Request form: {request.form}")
    
    try:
        if 'file' not in request.files:
            print("ERROR: Nema 'file' u request.files!")
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        print(f"Filename: {file.filename}")
        print(f"Content-Type: {file.content_type}")
        
        # Spremi fajl BEZ suffiksa
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            print(f"Fajl sparen: {tmp.name}")
            print(f"Veličina: {os.path.getsize(tmp.name)} bytes")
            
            try:
                print("Pokrećem Whisper...")
                result = model.transcribe(tmp.name, language="sr")
                os.unlink(tmp.name)
                print("USPJEH!")
                
                return jsonify({"text": result["text"]})
            except Exception as e:
                print(f"GREŠKA: {str(e)}")
                os.unlink
