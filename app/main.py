import streamlit as st
import whisper
import tempfile

st.title("🎧 Whisper Transcriber")
st.write("Upload an audio file and get the transcription using OpenAI's Whisper!")

uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    model = whisper.load_model("base")
    result = model.transcribe(tmp_path)
    
    st.subheader("📝 Transcription:")
    st.write(result["text"])
