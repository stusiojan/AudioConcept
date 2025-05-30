import torch
import typer
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix
from AudioConcept.models.model_cnn import CNN
from AudioConcept.models.model_vggish import VGGish
from AudioConcept.models.classifier_svm import SVMClassifier
from AudioConcept.dataset import get_data_loaders, gtzan_features_data, AudioLength
from AudioConcept.config import (
    MODEL_TO_TRAIN,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    GTZAN_GENRES,
)
from datetime import datetime

app = typer.Typer()

"""
Evaluate a trained model on the GTZAN dataset.
Saves results to the specified directories.
"""


@app.command()
def main(
    model_to_train: str = typer.Argument(default=MODEL_TO_TRAIN),
    model_path: str = MODELS_DIR,
    report_path: str = REPORTS_DIR,
    figures_path: str = FIGURES_DIR,
    audio_length: str = typer.Option(
        None,
        help="Audio length to use for training (CNN or VGG)",
    ),
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if audio_length is None:
        audio_length = AudioLength.CNN
    else:
        try:
            audio_length = AudioLength[audio_length.upper()]
        except KeyError:
            logger.error(
                f"Invalid audio length name: {audio_length}. Must be 'CNN' or 'VGG'."
            )
            raise ValueError(
                f"Invalid audio length name: {audio_length}. Must be 'CNN' or 'VGG'."
            )
    _, _, test_loader = get_data_loaders(audio_length)

    match model_to_train:
        case "VGGish":
            model = VGGish().to(device)
        case "CNN":
            model = CNN().to(device)
        case "SVM":
            logger.info(f"Loading best model for {model_to_train}...")
            classifier = SVMClassifier()
            classifier.load_model(f"{model_path}/best_{model_to_train}_model.pkl")
            _, X_test, _, y_test = gtzan_features_data()
            logger.info("Evaluating model...")
            accuracy, report, conf_mat = classifier.evaluate(X_test, y_test)
            logger.info(f"Test accuracy: {accuracy:.4f}")
            logger.info("\nClassification Report:")
            logger.info(report)

            classifier.plot_confusion_matrix(
                conf_mat,
                f"{figures_path}/{model_to_train}_confusion_matrix_{timestamp}.png",
            )
            return
        case _:
            logger.error(f"Unknown model: {model_to_train}")
            raise ValueError(f"Unknown model: {model_to_train}")

    logger.info(f"Loading best model for {model_to_train}...")

    with open(f"{model_path}/best_{model_to_train}_model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info(
        f"Loaded model from pickle file: {model_path}/best_{model_to_train}_model.pkl"
    )
    model = model.to(device)

    logger.info("Evaluating model...")
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for wav, genre_index in test_loader:
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            b, c, t = wav.size()
            logits = model(wav.view(-1, t))
            logits = logits.view(b, c, -1).mean(dim=1)
            _, pred = torch.max(logits.data, 1)

            y_true.extend(genre_index.tolist())
            y_pred.extend(pred.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    conf_matrix = sns.heatmap(
        cm,
        annot=True,
        xticklabels=GTZAN_GENRES,
        yticklabels=GTZAN_GENRES,
        cmap="YlGnBu",
    )
    plt.savefig(
        f"{figures_path}/{model_to_train}_confusion_matrix_{timestamp}_accu_{accuracy}.png",
        dpi=300,
        bbox_inches="tight",
    )
    logger.info("Accuracy: %.4f" % accuracy)

    return


if __name__ == "__main__":
    app()
