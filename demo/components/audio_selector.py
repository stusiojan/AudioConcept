import os
import streamlit as st

from AudioConcept.config import SAMPLE_AUDIO_DIR


def select_audio_file():
    audio_files = [f for f in os.listdir(SAMPLE_AUDIO_DIR) if f.endswith(".wav")]

    selected_file = st.selectbox("Choose sample:", audio_files)

    file_path = os.path.join(SAMPLE_AUDIO_DIR, selected_file)
    return file_path
