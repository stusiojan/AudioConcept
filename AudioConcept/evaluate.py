import torch
import typer
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix
from AudioConcept.modeling.model_cnn import CNN
from AudioConcept.modeling.model_vggish import VGGish
from AudioConcept.modeling.classifier_svm import SVMClassifier
from AudioConcept.dataset import test_loader, gtzan_features_data
from AudioConcept.config import (
    MODEL_TO_TRAIN,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    GTZAN_GENRES,
)
from datetime import datetime

app = typer.Typer()


@app.command()
def main(
    model_to_train: str = typer.Argument(default=MODEL_TO_TRAIN),
    model_path: str = MODELS_DIR,
    report_path: str = REPORTS_DIR,
    figures_path: str = FIGURES_DIR,
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
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
    S = torch.load(f"models/best_{model_to_train}_model.ckpt")
    model.load_state_dict(S)
    logger.info("loaded!")

    logger.info("Evaluating model...")
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for wav, genre_index in test_loader:
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            # reshape and aggregate chunk-level predictions
            b, c, t = wav.size()
            logits = model(wav.view(-1, t))
            logits = logits.view(b, c, -1).mean(dim=1)
            _, pred = torch.max(logits.data, 1)

            # append labels and predictions
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
        f"{figures_path}/{model_to_train}_confusion_matrix_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )
    logger.info("Accuracy: %.4f" % accuracy)

    return


if __name__ == "__main__":
    app()
