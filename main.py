from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import tempfile
import os
import time

app = Flask(__name__)

print("Loading model...")
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)
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
            start = time.time()

            segments, info = model.transcribe(
                tmp.name,
                language="sr",
                beam_size=1
            )

            text = " ".join(
                segment.text
                for segment in segments
            )

            print(
                f"Finished in {time.time() - start:.2f}s"
            )

            os.unlink(tmp.name)

            return jsonify({"text": text})

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
