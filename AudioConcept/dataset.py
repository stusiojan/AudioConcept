import os
import random
import torch
import typer
import numpy as np
import soundfile as sf
import pandas as pd
import librosa  # Add missing import
from pathlib import Path
from torch.utils import data
from AudioConcept.modeling.augmentation import (
    Compose,
    RandomResizedCrop,
    RandomApply,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    TimeStretch,
    SpectralRolloff,
)
from enum import Enum
from sklearn.model_selection import train_test_split
from AudioConcept.config import (
    DATA_PATH,
    PROCESSED_DATA_DIR,
    GTZAN_GENRES,
    SVM_TEST_SIZE,
    SVM_RANDOM_STATE,
)

app = typer.Typer()


class AudioLength(Enum):
    CNN = 22050 * 29.1
    VGG = 22050 * 3.96


class GTZANDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        split,
        num_samples,
        num_chunks,
        is_augmentation,
        sample_rate=22050,
    ):
        self.data_path = data_path if data_path else ""
        self.split = split
        self.num_samples = num_samples
        self.num_chunks = num_chunks
        self.is_augmentation = is_augmentation
        self.sample_rate = sample_rate
        self.genres = GTZAN_GENRES
        self._get_song_list()
        if is_augmentation:
            self._get_augmentations_enhanced()

    def _get_song_list(self):
        list_filename = os.path.join(self.data_path, "%s_filtered.txt" % self.split)
        with open(list_filename) as f:
            lines = f.readlines()
        self.song_list = [line.strip() for line in lines]

    def _get_augmentations(self):
        # Reduced and optimized augmentation pipeline to prevent memory issues
        transforms = [
            RandomResizedCrop(n_samples=self.num_samples),
            RandomApply([PolarityInversion()], p=0.5),  # Reduced probability
            RandomApply(
                [Noise(min_snr_db=15, max_snr_db=35)], p=0.3
            ),  # Reduced probability
            RandomApply([Gain(min_gain_db=-3, max_gain_db=3)], p=0.3),
            RandomApply(
                [HighLowPass(sample_rate=self.sample_rate)], p=0.4
            ),  # Reduced probability
            # Removed heavy augmentations for training stability
            # RandomApply([Delay(sample_rate=self.sample_rate)], p=0.2),
            # RandomApply([PitchShift(n_samples=self.num_samples, sample_rate=self.sample_rate)], p=0.2),
            # RandomApply([TimeStretch(n_samples=self.num_samples)], p=0.2),
            # RandomApply([Reverb(sample_rate=self.sample_rate)], p=0.2),
            # RandomApply([SpectralRolloff()], p=0.1),
        ]
        self.augmentation = Compose(transforms=transforms)

    def _get_augmentations_enhanced(self):
        """Enhanced augmentation pipeline to combat overfitting"""
        transforms = [
            # Basic augmentations (always applied)
            RandomResizedCrop(n_samples=self.num_samples),
            # Geometric augmentations
            RandomApply([PolarityInversion()], p=0.5),
            # Noise augmentations (help with robustness)
            RandomApply([Noise(min_snr_db=15, max_snr_db=35)], p=0.6),
            RandomApply([Gain(min_gain_db=-4, max_gain_db=4)], p=0.4),
            # Frequency domain augmentations
            RandomApply([HighLowPass(sample_rate=self.sample_rate)], p=0.5),
            # Time domain augmentations (add these back gradually)
            RandomApply([Delay(sample_rate=self.sample_rate)], p=0.3),
            # Advanced augmentations (use carefully)
            # RandomApply(
            #     [PitchShift(n_samples=self.num_samples, sample_rate=self.sample_rate)],
            #     p=0.2,
            # ),
            RandomApply([TimeStretch(n_samples=self.num_samples)], p=0.2),
        ]
        self.augmentation = Compose(transforms=transforms)

    def _adjust_audio_length(self, wav):
        """
        Random chunks of audio are cropped from the entire sequence during the
        training. But in validation / test phase, an entire sequence is split
        into multiple chunks and the chunks are stacked. The stacked chunks are
        later input to a trained model and the output predictions are aggregated
        to make song-level predictions.
        """
        if self.split == "train":
            if len(wav) <= self.num_samples:
                # Pad if too short
                pad_length = self.num_samples - len(wav)
                wav = np.pad(wav, (0, pad_length), mode="constant")
            else:
                # Random crop
                random_index = random.randint(0, len(wav) - self.num_samples)
                wav = wav[random_index : random_index + self.num_samples]
        else:
            if len(wav) <= self.num_samples:
                # Pad and create single chunk
                pad_length = self.num_samples - len(wav)
                wav = np.pad(wav, (0, pad_length), mode="constant")
                wav = np.array([wav])
            else:
                hop = max(1, (len(wav) - self.num_samples) // self.num_chunks)
                wav = np.array(
                    [
                        wav[i * hop : i * hop + self.num_samples]
                        for i in range(self.num_chunks)
                    ]
                )
        return wav

    def __getitem__(self, index):
        try:
            line = self.song_list[index]

            # get genre
            genre_name = line.split("/")[0]
            genre_index = self.genres.index(genre_name)

            # get audio
            audio_filename = os.path.join(self.data_path, "genres", line)

            # Use soundfile for loading (more reliable)
            try:
                wav, fs = sf.read(audio_filename)
            except Exception as e:
                print(f"Error loading {audio_filename} with soundfile: {e}")
                # Fallback to librosa
                wav, fs = librosa.load(audio_filename, sr=self.sample_rate, mono=True)

            # Ensure mono and correct sample rate
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)  # Convert to mono

            # Resample if necessary
            if fs != self.sample_rate:
                wav = librosa.resample(wav, orig_sr=fs, target_sr=self.sample_rate)

            # adjust audio length
            wav = self._adjust_audio_length(wav).astype("float32")

            # data augmentation - FIXED: Apply to numpy array directly
            if self.is_augmentation and self.split == "train":
                # Only apply augmentation to 1D audio (training phase)
                if wav.ndim == 1:
                    wav = self.augmentation(wav, self.sample_rate)
                    wav = wav.astype("float32")

            return wav, genre_index

        except Exception as e:
            print(f"Error processing sample {index}: {e}")
            # Return zeros as fallback
            if self.split == "train":
                return np.zeros(self.num_samples, dtype=np.float32), 0
            else:
                return (
                    np.zeros((self.num_chunks, self.num_samples), dtype=np.float32),
                    0,
                )

    def __len__(self):
        return len(self.song_list)

    def load_audio(self, file_path):
        """Load audio using librosa"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.zeros(self.num_samples)


def get_dataloader(
    data_path=DATA_PATH,
    split="train",
    audio_length=AudioLength.CNN,
    num_chunks=1,
    batch_size=16,
    num_workers=0,
    is_augmentation=False,
):
    num_samples = int(audio_length.value)
    is_shuffle = True if (split == "train") else False
    batch_size = batch_size if (split == "train") else (batch_size // num_chunks)

    # Reduce num_workers for augmented datasets to prevent memory issues
    if is_augmentation:
        num_workers = min(num_workers, 2)

    data_loader = data.DataLoader(
        dataset=GTZANDataset(
            data_path, split, num_samples, num_chunks, is_augmentation
        ),
        batch_size=batch_size,
        shuffle=is_shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,  # Disable to reduce memory usage
    )
    return data_loader


def gtzan_features_data(
    features_path: Path = PROCESSED_DATA_DIR / "processed_dataset.csv",
):

    def _get_features(processed_data):
        """Load data from processed dataset."""
        df = pd.read_csv(processed_data)
        X = df.drop("Y", axis=1)
        y = df["Y"]
        return X, y

    X, y = _get_features(features_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SVM_TEST_SIZE, random_state=SVM_RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def test_augmentation():
    print("Testing librosa-based audio augmentations...")

    # Generate dummy audio or load a real file
    sample_rate = 22050
    duration = 3  # seconds
    # Create a more realistic test signal (sine wave + noise)
    t = np.linspace(0, duration, sample_rate * duration)
    audio = (np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))).astype(
        np.float32
    )

    print(f"Original audio shape: {audio.shape}")
    print(f"Original audio RMS: {np.sqrt(np.mean(audio**2)):.4f}")

    # Create lightweight augmentation pipeline for testing
    augmentations = Compose(
        [
            RandomResizedCrop(n_samples=sample_rate * 3),
            RandomApply([Noise(min_snr_db=20, max_snr_db=40)], p=0.7),
            RandomApply([Gain(min_gain_db=-2, max_gain_db=2)], p=0.5),
            RandomApply([HighLowPass(sample_rate=sample_rate)], p=0.5),
        ]
    )

    # Apply augmentations multiple times to see variety
    for i in range(3):
        augmented_audio = augmentations(audio.copy(), sample_rate)
        print(
            f"Augmented audio {i+1} - Shape: {augmented_audio.shape}, RMS: {np.sqrt(np.mean(augmented_audio**2)):.4f}"
        )

    print("Augmentation pipeline test completed successfully!")


# Create data loaders
train_loader = get_dataloader(
    split="train",
    audio_length=AudioLength.CNN,
    is_augmentation=True,
    batch_size=6,
    num_workers=0,
)
valid_loader = get_dataloader(
    split="valid",
    audio_length=AudioLength.CNN,
    is_augmentation=False,
    batch_size=16,
    num_workers=2,
)
test_loader = get_dataloader(
    split="test",
    audio_length=AudioLength.CNN,
    is_augmentation=False,
    batch_size=16,
    num_workers=2,
)


@app.command()
def main():
    try:
        iter_train_loader = iter(train_loader)
        train_wav, train_genre = next(iter_train_loader)
        iter_test_loader = iter(test_loader)
        test_wav, test_genre = next(iter_test_loader)

        # Test features loading
        try:
            gtzan_features_data()
        except FileNotFoundError:
            print("Features file not found, skipping feature loading test")

        print("Training data shape: %s" % str(train_wav.shape))
        print("Training targets: ", train_genre)
        print("Validation/test data shape: %s" % str(test_wav.shape))
        print("Validation/test targets: ", test_genre)

        # Test augmentation
        test_augmentation()

        # Test with augmentation enabled
        print("\nTesting with augmentation enabled...")
        augmented_loader = get_dataloader(
            split="train",
            audio_length=AudioLength.CNN,
            is_augmentation=True,
            batch_size=4,  # Even smaller batch size for augmented data
            num_workers=0,  # Single threaded for stability
        )

        iter_augmented_loader = iter(augmented_loader)
        aug_wav, aug_genre = next(iter_augmented_loader)
        print("Augmented training data shape: %s" % str(aug_wav.shape))
        print("Augmented training targets: ", aug_genre)

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    app()
