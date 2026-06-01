from faster_whisper import WhisperModel
import time

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

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

return jsonify({"text": text})
