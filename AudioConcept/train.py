from pathlib import Path
from loguru import logger
from tqdm import tqdm

from config import (
    MODELS_DIR,
    FIGURES_DIR,
    MODEL_TO_TRAIN,
    LEARNING_RATE,
    NUM_EPOCHS,
    SVM_RANDOM_STATE,
)
import typer
import numpy as np
import torch
from torch import nn
from modeling.model_cnn import CNN
from modeling.model_vggish import VGGish
from modeling.classifier_svm import SVMClassifier
from dataset import train_loader, valid_loader, gtzan_features_data
from sklearn.metrics import accuracy_score
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

app = typer.Typer()


@app.command()
def main(
    model_to_train: str = MODEL_TO_TRAIN,
    model_path: Path = MODELS_DIR,
):
    logger.info(f"Training {MODEL_TO_TRAIN}...")
    train_model(model_to_train, model_path)
    logger.success("Training complete.")


def train_model(
    model_to_train: str,
    model_path: Path,
):
    # Tensorboard setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = f"GTZAN_{model_to_train}_{timestamp}_LR_{LEARNING_RATE}"
    writer = SummaryWriter(f"runs/{experiment_name}")

    # model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    match model_to_train:
        case "VGGish":
            model = VGGish().to(device)
        case "CNN":
            model = CNN().to(device)
        case "SVM":
            classifier = SVMClassifier(
                experiment_name="svm_genre_classifier", use_wandb=True
            )
            X_train, _, y_train, _ = gtzan_features_data()

            logger.info("Training model...")
            best_score = classifier.train(
                f"{model_path}/best_{model_to_train}_model.pkl",
                X_train,
                y_train,
                SVM_RANDOM_STATE,
            )
            logger.info(f"Best cross-validation score: {best_score:.4f}")
            return
        case _:
            logger.error(f"Unknown model: {model_to_train}")
            raise ValueError(f"Unknown model: {model_to_train}")
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    valid_losses = []
    num_epochs = NUM_EPOCHS

    for epoch in range(num_epochs):
        losses = []

        # train
        logger.info("Training model...")
        model.train()
        for batch_idx, (wav, genre_index) in enumerate(train_loader):
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            # forward
            out = model(wav)
            loss = loss_function(out, genre_index)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Training Loss", loss.item(), global_step)

            losses.append(loss.item())

        logger.info(
            "Epoch: [%d/%d], Train loss: %.4f"
            % (epoch + 1, num_epochs, np.mean(losses))
        )

        # Validation
        model.eval()
        y_true = []
        y_pred = []
        losses = []
        for wav, genre_index in valid_loader:
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            # reshape and aggregate chunk-level predictions
            b, c, t = wav.size()
            logits = model(wav.view(-1, t))
            logits = logits.view(b, c, -1).mean(dim=1)
            loss = loss_function(logits, genre_index)
            losses.append(loss.item())
            _, pred = torch.max(logits.data, 1)

            # append labels and predictions
            y_true.extend(genre_index.tolist())
            y_pred.extend(pred.tolist())
        accuracy = accuracy_score(y_true, y_pred)
        valid_loss = np.mean(losses)
        logger.info(
            "Epoch: [%d/%d], Valid loss: %.4f, Valid accuracy: %.4f"
            % (epoch + 1, num_epochs, valid_loss, accuracy)
        )

        # Save model
        valid_losses.append(valid_loss.item())
        if np.argmin(valid_losses) == epoch:
            logger.info("Saving the best model at %d epochs!" % epoch)
            torch.save(model_path, f"best_{model_to_train}_model.ckpt")

    writer.close()
    return


def aa(features_path, model_path, test_size=0.2, random_state=42, use_wandb=True):
    # """Train the SVM classifier.

    # Args:
    #     test_size: Proportion of dataset to include in the test split
    #     random_state: Random state for reproducibility
    #     use_wandb: Whether to use wandb logging
    # """
    # classifier = SVMClassifier(
    #     experiment_name="svm_genre_classifier", use_wandb=use_wandb
    # )

    # logger.info("Loading data...")
    # X, y = classifier.load_data(features_path)

    # logger.info("Splitting data...")
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size, random_state=random_state
    # )

    # logger.info("Training model...")
    # best_score = classifier.train(
    #     f"{model_path}/best_{model_to_train}_model.pkl", X_train, y_train, random_state
    # )  # +"svm_genre_classifier.pkl" do model_path
    # logger.info(f"Best cross-validation score: {best_score:.4f}")

    # might be moved to plots.py
    logger.info("Evaluating model...")
    accuracy, report, conf_mat = classifier.evaluate(X_test, y_test)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(report)

    classifier.plot_confusion_matrix(conf_mat, FIGURES_DIR / "confusion_matrix.png")

    # return classifier


if __name__ == "__main__":
    app()
