import sys
import os
from pathlib import Path

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.model_type import ModelType
from components.config import configure_page
from components.audio_selector import select_audio_file
from features.predictor import predict_genre_streamlit
from features.plot import generate_waterfall_plot
import streamlit as st

configure_page()

st.title("Music Genre Classifier with XAI")

possible_models = [model.value for model in ModelType]
model_choice = st.selectbox("Choose prediction model:", possible_models)

file_path = select_audio_file()

if file_path:
    st.audio(file_path)

    if st.button("Predict"):
        st.write("Prediction...")
        predict_genre_streamlit(file_path, model_choice)

    if st.button("Waterfall plot"):
        st.write("📊 Waterfall plot generating ...")
        fig = generate_waterfall_plot()
        st.pyplot(fig)
