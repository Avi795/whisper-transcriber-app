# app/audio_transcriber.py
import whisper

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file.name)
    return result["text"]

