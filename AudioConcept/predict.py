from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import numpy as np
import pandas as pd

from AudioConcept.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_AUDIO_DIR,
    MODEL_TO_TRAIN,
    REPORTS_DIR,
    GTZAN_GENRES,
)
from AudioConcept.modeling.classifier_svm import SVMClassifier

app = typer.Typer()


@app.command()
def main(
    prediction_model: str = typer.Argument(default=MODEL_TO_TRAIN),
    audio_to_predict: str = typer.Argument(
        default="test.wav",
        help="Path to the audio file to predict genre for.",
    ),
    audio_path: Path = SAMPLE_AUDIO_DIR,
    report_path: str = REPORTS_DIR,
    model_path: Path = MODELS_DIR,
):
    logger.info("Performing inference for model...")
    match prediction_model:
        case "SVM":
            pass
        case "VGGish":
            pass
        case "CNN":
            pass
        case _:
            logger.error(f"Unknown model: {prediction_model}")
            raise ValueError(f"Unknown model: {prediction_model}")
    logger.success("Inference complete.")


def validate_input_audio(audio_path: Path, audio_to_predict: str):
    pass


if __name__ == "__main__":
    app()
