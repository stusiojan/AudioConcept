import os
from pathlib import Path
import pickle

import librosa
from loguru import logger
import numpy as np
import soundfile as sf
import torch
import typer

from AudioConcept.config import (
    GTZAN_GENRES,
    MODEL_TO_TRAIN,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SAMPLE_AUDIO_DIR,
    VALIDATION_PARAMS,
)
from AudioConcept.dataset import calculate_features

app = typer.Typer()

"""
Command-line interface for performing genre prediction on audio files using trained models.
"""


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
        svm_scaler, model = load_model(prediction_model, model_path)
        probabilities = predict_genre(
            model, svm_scaler, audio_file_path, prediction_model
        )

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

    def _load_scaler():
        try:
            logger.info("Loading scaler...")
            with open(PROCESSED_DATA_DIR / "scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
                logger.info(
                    f" Scaler type: {type(scaler)}, values: {scaler.mean_}, {scaler.scale_}"
                )
            logger.success("Scaler loaded successfully")
        except FileNotFoundError:
            logger.error(
                "Scaler file not found. Please ensure the scaler.pkl file exists in the processed data directory."
            )
            raise
        return scaler

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
            else:
                raise ValueError(
                    f"Could not find SVM model in dictionary. Keys: {list(loaded_data.keys())}"
                )
            # if "scaler" in loaded_data:
            #     svm_scaler = loaded_data["scaler"]
            #     logger.success("SVM scaler loaded successfully")
            # else:
            #     raise ValueError(
            #         f"Could not find SVM scaler in dictionary. Keys: {list(loaded_data.keys())}"
            #     )
        else:
            # svm_scaler = None
            model = loaded_data
            logger.warning("Loaded data is not a dictionary.")
        svm_scaler = _load_scaler()
    else:
        svm_scaler = None
        model = loaded_data

    logger.success(f"Model {model_name} loaded successfully")
    return svm_scaler, model


def extract_features_for_svm(scaler, audio_file_path: Path) -> np.ndarray:
    """Extract features for SVM model."""
    audio_data, sample_rate = librosa.load(
        str(audio_file_path), sr=VALIDATION_PARAMS["target_sample_rate"]
    )
    features = calculate_features(audio_data, sample_rate, scaler)
    return features


def preprocess_audio_for_neural_models(audio_file_path: Path) -> np.ndarray:
    """Preprocess audio for neural network models."""
    audio_data, _ = librosa.load(
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


def predict_genre(
    model, svm_scaler, audio_file_path: Path, model_name: str
) -> np.ndarray:
    """Predict genre probabilities for the given audio file."""
    logger.info(f"Predicting genre using {model_name} model...")

    if model_name == "SVM":
        features = extract_features_for_svm(svm_scaler, audio_file_path)

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
