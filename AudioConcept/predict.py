from pathlib import Path
import pickle
import os

from loguru import logger
from tqdm import tqdm
import typer
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

from AudioConcept.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_AUDIO_DIR,
    MODEL_TO_TRAIN,
    REPORTS_DIR,
    GTZAN_GENRES,
    VALIDATION_PARAMS,
)
from AudioConcept.modeling.classifier_svm import SVMClassifier
from AudioConcept.modeling.model_cnn import CNN
from AudioConcept.modeling.model_vggish import VGGish

app = typer.Typer()


@app.command()
def main(
    prediction_model: str = typer.Argument(default=MODEL_TO_TRAIN),
    audio_path: Path = SAMPLE_AUDIO_DIR,
    report_path: str = REPORTS_DIR,
    model_path: Path = MODELS_DIR,
):
    audio_to_predict = prompt_user_choice()
    audio_file_path = audio_path / audio_to_predict

    logger.info(f"Performing inference for {prediction_model} model...")

    if not validate_input_audio(audio_file_path):
        logger.error("Audio validation failed. Exiting.")
        return

    try:
        model = load_model(prediction_model, model_path)
        probabilities = predict_genre(model, audio_file_path, prediction_model)

        logger.info("Genre prediction probabilities:")
        for i, genre in enumerate(GTZAN_GENRES):
            logger.info(f"{genre}: {probabilities[i]:.4f}")

        predicted_idx = np.argmax(probabilities)
        predicted_genre = GTZAN_GENRES[predicted_idx]
        logger.success(
            f"Predicted genre: {predicted_genre} (confidence: {probabilities[predicted_idx]:.4f})"
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

    logger.success("Inference complete.")


def prompt_user_choice() -> str:
    """
    Prompt user to select an audio file from the sample audio directory.
    """
    logger.info("Providing a list of available audio files for selection...")
    audio_files = [f for f in os.listdir(SAMPLE_AUDIO_DIR) if f.endswith(".wav")]
    if not audio_files:
        logger.error(
            "No audio files found in the sample audio directory."
            + " Add max 30 seconds mono .wav files with sample rate of 22050Hz to the 'sample_audio' directory."
        )
        raise FileNotFoundError("No audio files found.")

    print("Available audio files:")
    for i, file in enumerate(audio_files):
        print(f"{i + 1}: {file}")

    choice = typer.prompt(
        "Select an audio file by number",
        type=int,
        default=1,
        show_choices=False,
    )

    while choice < 1 or choice > len(audio_files):
        logger.error(
            f"Invalid choice: {choice}. Please select a number between 1 and {len(audio_files)}."
        )
        choice = typer.prompt(
            "Select an audio file by number",
            type=int,
            default=1,
            show_choices=False,
        )

    return audio_files[choice - 1]


def validate_input_audio(audio_file_path: Path) -> bool:
    """
    Validate input audio file according to requirements from VALIDATION_PARAMS
    """
    try:
        if not audio_file_path.exists():
            logger.error(f"Audio file not found: {audio_file_path}")
            return False

        if audio_file_path.suffix.lower() != VALIDATION_PARAMS["required_format"]:
            logger.error(
                f"Invalid format. Expected {VALIDATION_PARAMS['required_format']}, got {audio_file_path.suffix}"
            )
            return False

        try:
            audio_data, sample_rate = sf.read(str(audio_file_path))
        except Exception as e:
            logger.error(f"Failed to read audio file: {str(e)}")
            return False

        duration = len(audio_data) / sample_rate
        if duration >= VALIDATION_PARAMS["max_duration"]:
            logger.error(
                f"Audio duration ({duration:.2f}s) exceeds maximum ({VALIDATION_PARAMS['max_duration']}s)"
            )
            return False

        if (
            audio_data.ndim > 1
            and audio_data.shape[1] != VALIDATION_PARAMS["required_channels"]
        ):
            logger.error(f"Audio must be mono. Found {audio_data.shape[1]} channels")
            return False

        if sample_rate != VALIDATION_PARAMS["target_sample_rate"]:
            logger.warning(
                f"Sample rate is {sample_rate}Hz, expected {VALIDATION_PARAMS['target_sample_rate']}Hz. Will resample."
            )

        logger.success(
            f"Audio validation passed: {duration:.2f}s, {sample_rate}Hz, {audio_data.shape}"
        )
        return True

    except Exception as e:
        logger.error(f"Audio validation error: {str(e)}")
        return False


def load_model(model_name: str, model_path: Path):
    """Load trained model from pickle file."""
    model_file = model_path / f"best_{model_name}_model.pkl"

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    logger.info(f"Loading model from: {model_file}")

    with open(model_file, "rb") as f:
        loaded_data = pickle.load(f)

    if model_name == "SVM":
        if isinstance(loaded_data, dict):
            if "model" in loaded_data:
                model = loaded_data["model"]
            elif "best_model" in loaded_data:
                model = loaded_data["best_model"]
            elif "svm_model" in loaded_data:
                model = loaded_data["svm_model"]
            else:
                for key, value in loaded_data.items():
                    if hasattr(value, "predict") or hasattr(value, "decision_function"):
                        model = value
                        logger.info(f"Found model in key: {key}")
                        break
                else:
                    raise ValueError(
                        f"Could not find SVM model in dictionary. Keys: {list(loaded_data.keys())}"
                    )
        else:
            model = loaded_data
    else:
        model = loaded_data

    logger.success(f"Model {model_name} loaded successfully")
    return model


def extract_features_for_svm(audio_file_path: Path) -> np.ndarray:
    """Extract features for SVM model matching the training feature set (55 features)."""
    audio_data, sample_rate = librosa.load(
        str(audio_file_path), sr=VALIDATION_PARAMS["target_sample_rate"]
    )

    features = []

    length = len(audio_data) / sample_rate
    features.append(float(length))

    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    features.append(float(np.mean(chroma_stft)))
    features.append(float(np.var(chroma_stft)))

    rms = librosa.feature.rms(y=audio_data)
    features.append(float(np.mean(rms)))
    features.append(float(np.var(rms)))

    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
    features.append(float(np.var(spectral_centroids)))

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_data, sr=sample_rate
    )
    features.append(float(np.var(spectral_bandwidth)))

    rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
    features.append(float(np.mean(rolloff)))
    features.append(float(np.var(rolloff)))

    zcr = librosa.feature.zero_crossing_rate(audio_data)
    features.append(float(np.mean(zcr)))
    features.append(float(np.var(zcr)))

    harmony = librosa.effects.harmonic(audio_data)
    features.append(float(np.mean(harmony)))
    features.append(float(np.var(harmony)))

    tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sample_rate)
    features.append(float(np.mean(tonnetz)))
    features.append(float(np.var(tonnetz)))

    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
    features.append(float(tempo))

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)

    features.append(float(np.mean(mfccs[0])))
    features.append(float(np.var(mfccs[0])))
    features.append(float(np.var(mfccs[1])))

    # mfcc
    for i in range(2, 20):
        features.append(float(np.mean(mfccs[i])))
        features.append(float(np.var(mfccs[i])))

    features_array = np.array(features, dtype=np.float64)
    logger.info(f"Extracted {len(features_array)} features for SVM model")

    if len(features_array) != 55:
        logger.warning(f"Expected 55 features, got {len(features_array)}")
        logger.warning(f"Feature array shape: {features_array.shape}")
        logger.warning(f"Feature types: {[type(f) for f in features[:5]]}")

    return features_array.reshape(1, -1)


def preprocess_audio_for_neural_models(audio_file_path: Path) -> np.ndarray:
    """Preprocess audio for neural network models."""
    audio_data, sample_rate = librosa.load(
        str(audio_file_path), sr=VALIDATION_PARAMS["target_sample_rate"]
    )

    target_length = VALIDATION_PARAMS["target_sample_rate"] * 30
    if len(audio_data) < target_length:
        audio_data = np.pad(
            audio_data, (0, target_length - len(audio_data)), mode="constant"
        )
    else:
        audio_data = audio_data[:target_length]

    return audio_data


def predict_genre(model, audio_file_path: Path, model_name: str) -> np.ndarray:
    """Predict genre probabilities for the given audio file."""
    logger.info(f"Predicting genre using {model_name} model...")

    if model_name == "SVM":
        features = extract_features_for_svm(audio_file_path)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
        else:
            decision_scores = model.decision_function(features)[0]
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probabilities = exp_scores / np.sum(exp_scores)

    elif model_name in ["VGGish", "CNN"]:
        audio_data = preprocess_audio_for_neural_models(audio_file_path)

        audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0)

        device = next(model.parameters()).device
        logger.info(f"Model is on device: {device}")

        audio_tensor = audio_tensor.to(device)

        model.eval()

        with torch.no_grad():
            outputs = model(audio_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    else:
        raise ValueError(f"Unknown model type: {model_name}")

    return probabilities


if __name__ == "__main__":
    app()
