import streamlit as st
from pathlib import Path
from io import StringIO

from AudioConcept.predict import (
    validate_input_audio,
    load_model,
    predict_genre as raw_predict,
)
from AudioConcept.config import (
    SAMPLE_AUDIO_DIR,
    MODEL_TO_TRAIN,
    GTZAN_GENRES,
    MODELS_DIR,
)


class StreamlitLogHandler:
    """Redirects loguru logs to Streamlit with filtering for technical messages."""
    def __init__(self):
        self.buffer = StringIO()
        self.ignored_phrases = [
            "Loading model",
            "Model loaded",
            "Predicting genre",
        ]

    def write(self, message):
        if message.strip() and not any(phrase in message for phrase in self.ignored_phrases):
            st.text(message.strip())

    def flush(self):
        pass


def predict_genre_streamlit(file_path: str, model_choice: str):
    st.write(f"🎵 Selected model: **{model_choice}**")

    # Redirect loguru logs to Streamlit
    from loguru import logger
    logger.remove()
    logger.add(StreamlitLogHandler(), level="INFO", format="{message}")

    audio_file_path = Path(file_path)

    if not validate_input_audio(audio_file_path):
        st.error("❌ Audio file validation failed.")
        return None

    try:
        model = load_model(model_choice, MODELS_DIR)
        probabilities = raw_predict(model, audio_file_path, model_choice)

        st.markdown("### 📈 Genre prediction probabilities:")
        for genre, prob in zip(GTZAN_GENRES, probabilities):
            st.write(f"- **{genre}**: {prob:.4f}")

        predicted_idx = probabilities.argmax()
        predicted_genre = GTZAN_GENRES[predicted_idx]

        st.success(
            f"🎧 **Predicted genre:** `{predicted_genre}` "
            f"(confidence: {probabilities[predicted_idx]:.4f})"
        )

        return predicted_genre

    except Exception as e:
        st.exception(e)
        return None
