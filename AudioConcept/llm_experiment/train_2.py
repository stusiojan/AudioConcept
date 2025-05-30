import argparse
import json
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from AudioConcept.llm_experiment.dataset_2 import get_improved_dataloader
from config import DATA_PATH, GTZAN_GENRES

warnings.filterwarnings("ignore")
"""
LLM generated training script for GTZAN dataset classification.
"""


class MelSpectrogramCNN(nn.Module):
    """CNN Architecture optimized for mel-spectrogram classification"""

    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(MelSpectrogramCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Conv Block 1: 128x1292 -> 64x646
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Increased filters
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64, 64, kernel_size=3, padding=1
            ),  # Added second conv for better feature extraction
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # Conv Block 2: 64x646 -> 32x323
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Increased filters
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Added second conv
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            # Conv Block 3: 32x323 -> 16x161
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Increased filters
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Added second conv
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            # Conv Block 4: 16x161 -> 8x80
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Increased filters
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Added second conv
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),  # Increased unit count
            nn.BatchNorm1d(256),  # Added batch norm
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Added batch norm
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

        # Apply proper weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights to improve training convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization for Conv2d (ReLU)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for fully connected layers
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features
        x = self.conv_layers(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)
        return x


class GTZANTrainer:
    """Complete training pipeline for GTZAN classification"""

    def __init__(self, args):
        self.args = args

        # Use CUDA if available, otherwise fall back to MPS or CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # GTZAN genres
        self.genres = GTZAN_GENRES

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize model with improved architecture
        self.model = MelSpectrogramCNN(
            num_classes=len(self.genres), dropout_rate=args.dropout
        ).to(self.device)

        # Use label smoothing to improve robustness (helps with overfitting)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Better optimizer settings
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),  # Default Adam betas
            eps=1e-8,  # Default epsilon
        )

        # Improved learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,  # Reduce LR by half
            patience=args.patience,
            verbose=True,
            min_lr=args.min_lr
            / 10,  # Set minimum LR lower than early stopping threshold
        )

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": [],
        }

        self.best_val_acc = 0.0

        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def create_data_loaders(self):
        """Create train, validation, and test data loaders"""

        # Training loader with augmentation
        self.train_loader = get_improved_dataloader(
            data_path=self.args.data_path,
            split="train",
            genres=self.genres,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            augment=True,
        )

        # Validation loader
        self.val_loader = get_improved_dataloader(
            data_path=self.args.data_path,
            split="valid",
            genres=self.genres,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            augment=False,
        )

        # Test loader with multi-chunk processing - reduced number of chunks
        self.test_loader = get_improved_dataloader(
            data_path=self.args.data_path,
            split="test",
            genres=self.genres,
            batch_size=self.args.batch_size,  # Use normal batch size
            num_workers=self.args.num_workers,
            augment=False,
            num_chunks=1,  # Reduce chunks to 1 for now
        )

        print("Data loaded:")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Validation batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{100.0 * correct / total:.2f}%",
                    }
                )

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def test_model(self):
        """Test the model with simplified chunk processing"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            for data, target in pbar:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)

                # Get model predictions
                output = self.model(data)

                # Get probabilities
                probs = softmax(output)

                # Get predicted classes
                _, predicted = torch.max(probs, 1)

                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Calculate accuracy
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets)) * 100

        # Generate detailed classification report
        report = classification_report(
            all_targets, all_predictions, target_names=self.genres, output_dict=True
        )

        return accuracy, report, all_predictions, all_targets

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "args": vars(self.args),
        }

        # Save latest checkpoint
        torch.save(
            checkpoint, os.path.join(self.args.output_dir, "latest_checkpoint.pth")
        )

        # Save best checkpoint
        if is_best:
            torch.save(
                checkpoint, os.path.join(self.args.output_dir, "best_checkpoint.pth")
            )
            print(
                f"New best model saved with validation accuracy: {self.best_val_acc:.2f}%"
            )

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_acc = checkpoint["best_val_acc"]
        self.history = checkpoint["history"]

        return checkpoint["epoch"]

    def plot_training_history(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(self.history["train_loss"], label="Training Loss", color="blue")
        ax1.plot(self.history["val_loss"], label="Validation Loss", color="red")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(self.history["train_acc"], label="Training Accuracy", color="blue")
        ax2.plot(self.history["val_acc"], label="Validation Accuracy", color="red")
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.args.output_dir, "training_curves.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_confusion_matrix(self, predictions, targets):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.genres,
            yticklabels=self.genres,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.args.output_dir, "confusion_matrix.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def cleanup_dataloader_resources(self):
        """Explicitly cleanup DataLoader resources to prevent file descriptor leaks"""
        for dataloader_name in ["train_loader", "val_loader", "test_loader"]:
            if hasattr(self, dataloader_name):
                dataloader = getattr(self, dataloader_name)
                if hasattr(dataloader, "_iterator"):
                    del dataloader._iterator
                setattr(self, dataloader_name, None)

        # Force garbage collection
        import gc

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("DataLoader resources cleaned up")

    def train(self):
        """Main training loop"""
        print("Starting training...")
        start_time = time.time()

        # Create data loaders
        self.create_data_loaders()

        # Resume from checkpoint if specified
        start_epoch = 0
        if self.args.resume:
            start_epoch = self.load_checkpoint(self.args.resume) + 1
            print(f"Resumed training from epoch {start_epoch}")

        try:
            # Training loop
            for epoch in range(start_epoch, self.args.epochs):
                print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
                print("-" * 50)

                # Train
                train_loss, train_acc = self.train_epoch()

                # Validate
                val_loss, val_acc = self.validate_epoch()

                # Update learning rate
                self.scheduler.step(val_acc)
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Update history
                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["learning_rates"].append(current_lr)

                # Print epoch results
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                print(f"Learning Rate: {current_lr:.6f}")

                # Save checkpoint
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc

                self.save_checkpoint(epoch, is_best)

                # Close dataloaders periodically to prevent file descriptor leaks
                if epoch > 0 and epoch % 5 == 0:
                    print("Cleaning up DataLoader resources...")
                    self.cleanup_dataloader_resources()
                    # Recreate data loaders
                    self.create_data_loaders()

                # Early stopping
                if current_lr < self.args.min_lr:
                    print(
                        f"Learning rate {current_lr} below minimum {self.args.min_lr}. Stopping training."
                    )
                    break

        except Exception as e:
            print(f"Error during training: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # Always cleanup resources
            self.cleanup_dataloader_resources()

        # Training completed
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time / 3600:.2f} hours")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        # Plot training curves
        self.plot_training_history()

        # Test the best model
        print("\nTesting best model...")
        self.load_checkpoint(os.path.join(self.args.output_dir, "best_checkpoint.pth"))

        # Recreate dataloaders for testing only
        self.create_data_loaders()
        test_acc, test_report, predictions, targets = self.test_model()
        self.cleanup_dataloader_resources()  # Clean up after testing

        print(f"Test Accuracy: {test_acc:.2f}%")
        print("\nDetailed Classification Report:")
        print(classification_report(targets, predictions, target_names=self.genres))

        # Save results
        results = {
            "test_accuracy": test_acc,
            "best_val_accuracy": self.best_val_acc,
            "training_time_hours": training_time / 3600,
            "classification_report": test_report,
        }

        with open(os.path.join(self.args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Plot confusion matrix
        self.plot_confusion_matrix(predictions, targets)

        print(f"\nAll results saved to: {self.args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="GTZAN Music Genre Classification Training"
    )

    # Data parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help="Path to GTZAN dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results and checkpoints",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    # Scheduler parameters
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for learning rate scheduler"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for early stopping",
    )

    # Data loading parameters
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of data loading workers"
    )
    parser.add_argument(
        "--test_chunks",
        type=int,
        default=5,
        help="Number of chunks for test-time processing",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    args = parser.parse_args()

    # Create trainer and start training
    trainer = GTZANTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
