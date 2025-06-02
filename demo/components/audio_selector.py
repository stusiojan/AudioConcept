import os
import streamlit as st
from pathlib import Path

# Use absolute path to the audio directory
AUDIO_DIR = os.path.join(Path(__file__).parent.parent, "audio", "genres_original")


def select_audio_file():
    genres = [
        d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))
    ]
    selected_genre = st.selectbox("Choose music genre:", genres)

    genre_path = os.path.join(AUDIO_DIR, selected_genre)
    audio_files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
    selected_file = st.selectbox("Choose sample:", audio_files)

    file_path = os.path.join(genre_path, selected_file)
    return file_path
