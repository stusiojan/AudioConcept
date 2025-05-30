from pathlib import Path
from loguru import logger
from tqdm import tqdm
import pickle

from AudioConcept.config import (
    MODELS_DIR,
    FIGURES_DIR,
    MODEL_TO_TRAIN,
    LEARNING_RATE,
    NUM_EPOCHS,
    SVM_RANDOM_STATE,
    MODEL_PATIENCE,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    NOISE_LEVEL,
)
import typer
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import json
from AudioConcept.modeling.model_cnn import CNN
from AudioConcept.modeling.model_cnn2 import CNN2
from AudioConcept.modeling.model_vggish import VGGish
from AudioConcept.modeling.classifier_svm import SVMClassifier
from AudioConcept.dataset import get_data_loaders, gtzan_features_data, AudioLength
from sklearn.metrics import accuracy_score
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

app = typer.Typer()


@app.command()
def main(
    model_to_train: str = typer.Argument(default=MODEL_TO_TRAIN),
    model_path: Path = MODELS_DIR,
    figures_path: Path = FIGURES_DIR,
    audio_length: str = typer.Option(
        None,
        help="Audio length to use for training (CNN or VGG)",
    ),
):
    logger.info(f"Training {model_to_train}...")
    train_model(
        model_to_train=model_to_train,
        model_path=model_path,
        figures_path=figures_path,
        audio_length=audio_length,
    )
    logger.success("Training complete.")


@app.command()
def experiment(
    model_to_train: str = typer.Argument(default=MODEL_TO_TRAIN),
    model_path: Path = MODELS_DIR,
    figures_path: Path = FIGURES_DIR,
    audio_length: str = typer.Option(
        None, help="Audio length to use for training (CNN or VGG)"
    ),
    lr: float = typer.Option(None, help="Learning rate to experiment with"),
    weight_decay: float = typer.Option(None, help="Weight decay to experiment with"),
    label_smoothing: float = typer.Option(
        None, help="Label smoothing to experiment with"
    ),
    noise_level: float = typer.Option(None, help="Noise level for data augmentation"),
):
    logger.info(f"Running experiment for {model_to_train}...")
    logger.info(
        f"Experiment params: LR={lr}, WD={weight_decay}, LS={label_smoothing}, Noise={noise_level}"
    )
    final_acc = train_model(
        model_to_train,
        model_path,
        figures_path,
        audio_length,
        experiment_lr=lr,
        experiment_weight_decay=weight_decay,
        experiment_label_smoothing=label_smoothing,
        experiment_noise_level=noise_level,
    )
    logger.success(f"Experiment complete. Final accuracy: {final_acc:.4f}")


def train_model(
    model_to_train: str,
    model_path: Path,
    figures_path: Path,
    audio_length: str = None,
    experiment_lr: float = None,
    experiment_weight_decay: float = None,
    experiment_label_smoothing: float = None,
    experiment_noise_level: float = None,
):
    # Tensorboard setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = f"GTZAN_{model_to_train}_{timestamp}_LR_{LEARNING_RATE}"
    writer = SummaryWriter(f"runs/{experiment_name}")

    # data loading
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
    train_loader, valid_loader, test_loader = get_data_loaders(audio_length)

    # model setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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

            # Save SVM stats
            run_stats = {
                "model": model_to_train,
                "best_score": best_score,
                "timestamp": timestamp,
            }
            with open(
                figures_path / f"{model_to_train}_run_stats_{timestamp}.json", "w"
            ) as f:
                json.dump(run_stats, f)
            logger.info(
                f"Saved run stats to {figures_path}/{model_to_train}_run_stats_{timestamp}.json"
            )
            return
        case _:
            logger.error(f"Unknown model: {model_to_train}")
            raise ValueError(f"Unknown model: {model_to_train}")

    # hyperparameters setup
    effective_lr = experiment_lr if experiment_lr else LEARNING_RATE
    weight_decay = experiment_weight_decay if experiment_weight_decay else WEIGHT_DECAY
    label_smoothing = (
        experiment_label_smoothing if experiment_label_smoothing else LABEL_SMOOTHING
    )
    noise_level = experiment_noise_level if experiment_noise_level else NOISE_LEVEL
    logger.info(
        f"Experiment parameters: LR={effective_lr}, WD={weight_decay}, LS={label_smoothing}, Noise={noise_level}"
    )

    loss_function = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=effective_lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.75,
        patience=8,
        min_lr=effective_lr / 50,
        verbose=True,
    )

    num_epochs = NUM_EPOCHS
    patience = MODEL_PATIENCE
    patience_counter = 0
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    best_valid_loss = float("inf")
    best_valid_accuracy = float("-inf")
    best_model_state = None

    for epoch in range(num_epochs):
        losses = []
        y_true_train = []
        y_pred_train = []
        logger.info(f"Training epoch {epoch + 1}/{num_epochs}...")
        model.train()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train")

        for batch_idx, (wav, genre_index) in enumerate(train_pbar):
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            if batch_idx == 0 and epoch == 0:
                logger.info(f"Input shape: {wav.shape}")
                logger.info(
                    f"Genre distribution in batch: {torch.bincount(genre_index)}"
                )
                logger.info(f"Input range: [{wav.min():.3f}, {wav.max():.3f}]")

            # Only for experiments with data augmentation
            if model.training and noise_level > 0:
                noise = torch.randn_like(wav) * noise_level
                wav = wav + noise

            # Forward pass
            out = model(wav)
            loss = loss_function(out, genre_index)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Training Loss", loss.item(), global_step)

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
            writer.add_scalar("Gradient Norm", total_norm, global_step)

            if batch_idx % 100 == 0:
                writer.add_scalar("Batch Training Loss", loss.item(), global_step)

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            losses.append(loss.item())
            _, pred = torch.max(out.data, 1)
            y_true_train.extend(genre_index.cpu().tolist())
            y_pred_train.extend(pred.cpu().tolist())

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = np.mean(losses)
        train_accuracy = accuracy_score(y_true_train, y_pred_train)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        y_true = []
        y_pred = []
        losses = []

        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1} Valid")

        with torch.no_grad():
            for wav, genre_index in valid_pbar:
                wav = wav.to(device)
                genre_index = genre_index.to(device)

                b, c, t = wav.size()
                logits = model(wav.view(-1, t))
                logits = logits.view(b, c, -1).mean(dim=1)
                loss = loss_function(logits, genre_index)
                losses.append(loss.item())
                _, pred = torch.max(logits.data, 1)

                y_true.extend(genre_index.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())

                valid_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        accuracy = accuracy_score(y_true, y_pred)
        valid_loss = np.mean(losses)
        valid_losses.append(valid_loss)
        valid_accuracies.append(accuracy)

        scheduler.step(accuracy)  # Now using accuracy for scheduler instead of loss

        writer.add_scalar("Validation Loss", valid_loss, epoch)
        writer.add_scalar("Validation Accuracy", accuracy, epoch)
        writer.add_scalar("Training Accuracy", train_accuracy, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            f"Epoch: [{epoch + 1}/{num_epochs}], "
            f"Train loss: {train_loss:.4f}, Train acc: {train_accuracy:.4f}, "
            f"Valid loss: {valid_loss:.4f}, Valid acc: {accuracy:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if accuracy > best_valid_accuracy:
            best_valid_accuracy = accuracy
            best_valid_loss = valid_loss  # Still track the loss at best accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info(
                f"New best validation accuracy: {best_valid_accuracy:.4f} - Saving model!"
            )
            # Save model checkpoint
            torch.save(
                model.state_dict(), f"{model_path}/best_{model_to_train}_model.ckpt"
            )

            # Save model as pickle file
            with open(f"{model_path}/best_{model_to_train}_model.pkl", "wb") as f:
                pickle.dump(model, f)
            logger.info(
                f"Saved model as pickle file to {model_path}/best_{model_to_train}_model.pkl"
            )
        else:
            patience_counter += 1
            logger.info(f"No improvement - patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model for final evaluation")

        # Double check that final model is saved as pickle
        with open(f"{model_path}/final_{model_to_train}_model.pkl", "wb") as f:
            pickle.dump(model, f)
        logger.info(
            f"Saved final model as pickle file to {model_path}/final_{model_to_train}_model.pkl"
        )

    model.eval()
    final_y_true, final_y_pred = [], []
    with torch.no_grad():
        for wav, genre_index in valid_loader:
            wav = wav.to(device)
            genre_index = genre_index.to(device)
            b, c, t = wav.size()
            logits = model(wav.view(-1, t))
            logits = logits.view(b, c, -1).mean(dim=1)
            _, pred = torch.max(logits.data, 1)
            final_y_true.extend(genre_index.cpu().tolist())
            final_y_pred.extend(pred.cpu().tolist())

    final_accuracy = accuracy_score(final_y_true, final_y_pred)
    logger.info(f"Final validation accuracy with best model: {final_accuracy:.4f}")

    # plotting and saving results
    figures_path.mkdir(parents=True, exist_ok=True)

    run_stats = {
        "model": model_to_train,
        "timestamp": timestamp,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "valid_losses": valid_losses,
        "valid_accuracies": valid_accuracies,
        "epochs": list(range(1, len(train_losses) + 1)),
        "final_accuracy": final_accuracy,
        "best_valid_accuracy": best_valid_accuracy,
        "best_valid_loss": best_valid_loss,
        "effective_learning_rate": effective_lr,
        "early_stopped_at_epoch": len(train_losses),
    }

    with open(figures_path / f"{model_to_train}_run_stats_{timestamp}.json", "w") as f:
        json.dump(run_stats, f, indent=2)
    logger.info(
        f"Saved run stats to {figures_path}/{model_to_train}_run_stats_{timestamp}.json"
    )

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, label="Training Loss", alpha=0.8)
    plt.plot(epochs_range, valid_losses, label="Validation Loss", alpha=0.8)

    best_epoch = np.argmax(valid_accuracies) + 1
    plt.axvline(
        x=best_epoch,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Best Epoch ({best_epoch})",
    )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_to_train} Loss (LR: {effective_lr:.1e})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label="Training Accuracy", alpha=0.8)
    plt.plot(epochs_range, valid_accuracies, label="Validation Accuracy", alpha=0.8)

    plt.axvline(
        x=best_epoch,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Best Epoch ({best_epoch})",
    )

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(
        f"{model_to_train} Accuracy (Best: {best_valid_accuracy:.3f}, Final: {final_accuracy:.3f})"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        figures_path / f"{model_to_train}_training_plots_{timestamp}.png", dpi=300
    )
    logger.info(
        f"Saved training plots to {figures_path}/{model_to_train}_training_plots_{timestamp}.png"
    )

    writer.close()
    return final_accuracy


if __name__ == "__main__":
    app()
