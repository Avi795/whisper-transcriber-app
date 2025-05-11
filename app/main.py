import os
import tempfile
import whisper
import streamlit as st

st.title("🎙 Whisper Audio Transcriber")

model = whisper.load_model("base")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.success("File uploaded successfully! Transcribing...")

        result = model.transcribe(tmp_path)

        st.subheader("Transcription:")
        st.write(result["text"])

        # Clean up temp file
        os.remove(tmp_path)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload an audio file to begin.")
