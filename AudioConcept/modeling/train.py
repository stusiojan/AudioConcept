from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
from sklearn.model_selection import train_test_split

from AudioConcept.config import MODELS_DIR, PROCESSED_DATA_DIR
from AudioConcept.modeling.svm_classifier import SVMClassifier

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "processed_dataset.csv",
    # labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "svm_genre_classifier.pkl",
):
    logger.info("Training SVM...")
    train_model(features_path, model_path)
    logger.success("Modeling training complete.")


def train_model(
    features_path, model_path, test_size=0.2, random_state=42, use_wandb=True
):
    """Train the SVM classifier.

    Args:
        test_size: Proportion of dataset to include in the test split
        random_state: Random state for reproducibility
        use_wandb: Whether to use wandb logging
    """
    classifier = SVMClassifier(
        experiment_name="svm_genre_classifier", use_wandb=use_wandb
    )

    logger.info("Loading data...")
    X, y = classifier.load_data(features_path)

    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info("Training model...")
    best_score = classifier.train(model_path, X_train, y_train, random_state)
    logger.info(f"Best cross-validation score: {best_score:.4f}")

    # might be moved to plots.py
    logger.info("Evaluating model...")
    accuracy, report, conf_mat = classifier.evaluate(X_test, y_test)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(report)

    classifier.plot_confusion_matrix(
        conf_mat, PROCESSED_DATA_DIR / "confusion_matrix.png"
    )

    # return classifier


if __name__ == "__main__":
    app()
