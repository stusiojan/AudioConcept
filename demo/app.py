import streamlit as st
import os
from XAI.featureExtractor import FeatureExtractor
from XAI.xaiWaterfall import xaiWaterfall
from AudioConcept.modeling.svm_classifier import SVMClassifier
from AudioConcept.config import PROCESSED_DATA_DIR, MODELS_DIR

# Ścieżka główna do folderu audio
AUDIO_DIR = "audio/genres_original"

# Listuj gatunki (foldery)
genres = [d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))]

# Selectbox do wyboru gatunku
selected_genre = st.selectbox("Choose music genre:", genres)

# Ścieżka do wybranego folderu
genre_path = os.path.join(AUDIO_DIR, selected_genre)

# Listuj pliki audio w wybranym gatunku
audio_files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]

# Selectbox do wyboru pliku audio
selected_file = st.selectbox("Choose sample:", audio_files)

# Ścieżka do wybranego pliku
file_path = os.path.join(genre_path, selected_file)

# Odtwarzanie audio
st.audio(file_path)

# Przycisk do wykonania predykcji
if st.button("Predict"):
    st.write("🔍 Przeprowadzanie ekstrakcji cech i predykcji...")

    # Ekstrakcja cech
    extractor = FeatureExtractor()
    df = extractor.extract_features(file_path)

    # Usuń mfcc2_mean jeśli istnieje
    if 'mfcc2_mean' in df.columns:
        df = df.drop(columns=['mfcc2_mean'])

    # Ładowanie modelu i predykcja
    model_path = MODELS_DIR / "svm_genre_classifier.pkl"
    classifier = SVMClassifier()
    classifier.load_model(model_path)
    prediction = classifier.predict(df.values)

    # Wyświetlenie wyniku
    st.success(f"🎵 Przewidywany gatunek muzyczny: **{prediction[0]}**")

# Waterfall plot
if st.button("Waterfall plot"):
    st.write("Generowanie wykresu waterfall...")
    xai = xaiWaterfall()
    xai.load_data()
    xai.load_model()
    xai.scale_data()
    xai.compute_shap_values()
    st.pyplot(xai.plot_waterfall())

