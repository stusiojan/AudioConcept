from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import numpy as np
import pandas as pd

from AudioConcept.config import MODELS_DIR, PROCESSED_DATA_DIR
from AudioConcept.modeling.svm_classifier import SVMClassifier

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv", # No data to predict
    model_path: Path = MODELS_DIR / "svm_genre_classifier.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    logger.info("Performing inference for model...")
    input_features = pd.read_csv(features_path)
    predictions = predict(model_path, input_features)
    predictions.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")
    logger.success("Inference complete.")


def predict(model_path, input_features, model_name="svm_genre_classifier"):
    """Make predictions using a trained SVM classifier.

    Args:
        input_features: DataFrame or array of features
        model_name: Name of the saved model to use

    Returns:
        Predicted genre labels
    """
    classifier = SVMClassifier()
    classifier.load_model(model_path)

    if isinstance(input_features, pd.DataFrame):
        X = input_features.values
    else:
        X = np.array(input_features)

    predictions = classifier.predict(X)
    logger.info(predictions)

    return predictions


if __name__ == "__main__":
    app()
