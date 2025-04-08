from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import numpy as np

from AudioConcept.config import FIGURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from AudioConcept.modeling.svm_classifier import SVMClassifier

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    conf_output_path: Path = FIGURES_DIR / "confusion_matrix.png",
    table_output_path: Path = FIGURES_DIR / "svm_results.png",
    model_path: Path = MODELS_DIR / "svm_genre_classifier.pkl",
):
    try:
        wandb.init(project="audio-concept", name="visualization")
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        logger.warning("Continuing without wandb logging...")

    # logger.info("Loading classifier")
    # classifier = SVMClassifier()
    # classifier.load_model(model_path)

    # logger.info("Generating plot from data...")
    # classifier.plot_confusion_matrix()

    logger.success("Plot generation complete.")


if __name__ == "__main__":
    app()
