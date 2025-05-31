from enum import Enum
import os
from pathlib import Path
import random

import librosa
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import soundfile as sf
from torch.utils import data
import typer

from AudioConcept.augmentation import (
    Compose,
    Delay,
    Gain,
    HighLowPass,
    Noise,
    PolarityInversion,
    RandomApply,
    RandomResizedCrop,
    TimeStretch,
)
from AudioConcept.config import (
    DATA_PATH,
    GTZAN_GENRES,
    PROCESSED_DATA_DIR,
    SVM_FEATURES_FILTER,
    SVM_RANDOM_STATE,
    SVM_TEST_SIZE,
)
from AudioConcept.features import AudioFeatureExtractor

app = typer.Typer()


"""
Prepare the GTZAN dataset for training, evaluation and testing.
"""


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
            self._get_augmentations()

    def _get_song_list(self):
        list_filename = os.path.join(self.data_path, "%s_filtered.txt" % self.split)
        with open(list_filename) as f:
            lines = f.readlines()
        self.song_list = [line.strip() for line in lines]

    def _get_augmentations(self):
        """Enhanced augmentation pipeline to combat overfitting"""
        transforms = [
            RandomResizedCrop(n_samples=self.num_samples),
            RandomApply([PolarityInversion()], p=0.5),
            RandomApply([Noise(min_snr_db=15, max_snr_db=35)], p=0.6),
            RandomApply([Gain(min_gain_db=-4, max_gain_db=4)], p=0.4),
            RandomApply([HighLowPass(sample_rate=self.sample_rate)], p=0.5),
            RandomApply([Delay(sample_rate=self.sample_rate)], p=0.3),
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
                pad_length = self.num_samples - len(wav)
                wav = np.pad(wav, (0, pad_length), mode="constant")
            else:
                random_index = random.randint(0, len(wav) - self.num_samples)
                wav = wav[random_index : random_index + self.num_samples]
        else:
            if len(wav) <= self.num_samples:
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

            genre_name = line.split("/")[0]
            genre_index = self.genres.index(genre_name)

            audio_filename = os.path.join(self.data_path, "genres", line)

            try:
                wav, fs = sf.read(audio_filename)
            except Exception as e:
                logger.error(f"Error loading {audio_filename} with soundfile: {e}")
                wav, fs = librosa.load(audio_filename, sr=self.sample_rate, mono=True)

            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)

            if fs != self.sample_rate:
                wav = librosa.resample(wav, orig_sr=fs, target_sr=self.sample_rate)

            wav = self._adjust_audio_length(wav).astype("float32")

            if self.is_augmentation and self.split == "train":
                if wav.ndim == 1:
                    wav = self.augmentation(wav, self.sample_rate)
                    wav = wav.astype("float32")

            return wav, genre_index

        except Exception as e:
            logger.error(f"Error processing sample {index}: {e}")
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
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
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
        pin_memory=False,
    )
    return data_loader


def gtzan_features_data(
    features_path: Path = PROCESSED_DATA_DIR / "processed_dataset.csv",
    feature_filter: list[str] = SVM_FEATURES_FILTER,
):
    """
    Preparing features for SVM model.
    Loads already processed dataset and filters it.
    Change filter in config.py to select features.
    """

    def _get_features(processed_data):
        """Load data from processed dataset."""
        df = pd.read_csv(processed_data)
        X = df.drop("Y", axis=1)
        y = df["Y"]
        return X, y

    def _filter_features(X, feature_filter):
        """Filter features based on the provided filter."""
        if feature_filter:
            filtered_cols = [col for col in X.columns if col in feature_filter]
            X = X[filtered_cols]
            logger.info(
                f"Applied feature filter: {len(filtered_cols)} features selected"
            )
            # X = X[feature_filter]
        return X

    X, y = _get_features(features_path)
    X = _filter_features(X, feature_filter)
    if X.empty or y.empty:
        raise ValueError(
            "Features or labels are empty. Check the processed dataset and feature filter."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SVM_TEST_SIZE, random_state=SVM_RANDOM_STATE
    )
    logger.info(f"Features shape: {X.shape}")

    return X_train, X_test, y_train, y_test


def calculate_features(
    audio_data: np.ndarray,
    sample_rate: int,
    svm_scaler=None,
    feature_filter: list[str] = SVM_FEATURES_FILTER,
    features_path: Path = PROCESSED_DATA_DIR / "processed_audio.csv",
):
    def _filter_features(features_dict, feature_filter):
        """Filter features based on the provided filter."""
        if feature_filter:
            return {k: v for k, v in features_dict.items() if k in feature_filter}
        return features_dict

    def _scale_features(features_array, scaler):
        """Scale features using the provided scaler."""
        if scaler is not None:
            return scaler.transform(features_array)
        logger.warning("No scaler provided, returning features without scaling.")
        features_df = pd.DataFrame(features_array, columns=features_array.columns)
        return features_df

    def _validate_values(features_dict):
        """Ensure all values in the features dictionary are scalar."""
        for key, value in features_dict.items():
            if not np.isscalar(value) and not isinstance(value, (int, float)):
                raise ValueError(f"Feature '{key}' has non-scalar value: {value}")

    extractor = AudioFeatureExtractor()
    features_dict = extractor.extract_features(audio_data, sample_rate)
    filtered_features_dict = _filter_features(features_dict, feature_filter)
    _validate_values(filtered_features_dict)
    filtered_values = list(filtered_features_dict.values())
    logger.info(f"Features: {list(filtered_features_dict.keys())}")

    features_array = np.array(filtered_values).reshape(1, -1)
    features_df = pd.DataFrame(
        features_array, columns=list(filtered_features_dict.keys())
    )
    filtered_features_df = _scale_features(features_df, svm_scaler)

    try:
        features_path.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(filtered_features_dict, index=[0]).to_csv(
            features_path, index=False
        )
        logger.info(f"Features saved to {features_path}")
    except Exception as e:
        logger.error(f"Failed to save features to {features_path}: {e}")

    # result = extractor.scale_features(features_path, features_path)

    return filtered_features_df


def test_augmentation():
    logger.info("Testing librosa-based audio augmentations...")

    # Dummy audio
    sample_rate = 22050
    duration = 3  # seconds
    t = np.linspace(0, duration, sample_rate * duration)
    audio = (np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))).astype(
        np.float32
    )

    logger.info(f"Original audio shape: {audio.shape}")
    logger.info(f"Original audio RMS: {np.sqrt(np.mean(audio**2)):.4f}")

    augmentations = Compose(
        [
            RandomResizedCrop(n_samples=sample_rate * 3),
            RandomApply([Noise(min_snr_db=20, max_snr_db=40)], p=0.7),
            RandomApply([Gain(min_gain_db=-2, max_gain_db=2)], p=0.5),
            RandomApply([HighLowPass(sample_rate=sample_rate)], p=0.5),
        ]
    )

    for i in range(3):
        augmented_audio = augmentations(audio.copy(), sample_rate)
        logger.info(
            f"Augmented audio {i + 1} - Shape: {augmented_audio.shape}, RMS: {np.sqrt(np.mean(augmented_audio**2)):.4f}"
        )

    logger.info("Augmentation pipeline test completed successfully!")


def get_data_loaders(audioLength: AudioLength = AudioLength.CNN):
    logger.info(f"Using audio length: {audioLength.name} : {audioLength.value} samples")
    train_loader = get_dataloader(
        split="train",
        audio_length=audioLength,
        is_augmentation=True,
        batch_size=6,
        num_workers=0,
    )
    valid_loader = get_dataloader(
        split="valid",
        audio_length=audioLength,
        is_augmentation=False,
        batch_size=16,
        num_workers=2,
    )
    test_loader = get_dataloader(
        split="test",
        audio_length=audioLength,
        is_augmentation=False,
        batch_size=16,
        num_workers=2,
    )
    return train_loader, valid_loader, test_loader


@app.command()
def main():
    try:
        train_loader, _, test_loader = get_data_loaders()

        iter_train_loader = iter(train_loader)
        train_wav, train_genre = next(iter_train_loader)
        iter_test_loader = iter(test_loader)
        test_wav, test_genre = next(iter_test_loader)

        try:
            gtzan_features_data()
        except FileNotFoundError:
            logger.error("Features file not found, skipping feature loading test")

        logger.info(f"Training data shape: {train_wav.shape}")
        logger.info(f"Training targets: {train_genre}")
        logger.info(f"Validation/test data shape: {test_wav.shape}")
        logger.info(f"Validation/test targets: {test_genre}")

        test_augmentation()

        logger.info("\nTesting with augmentation enabled...")
        augmented_loader = get_dataloader(
            split="train",
            audio_length=AudioLength.CNN,
            is_augmentation=True,
            batch_size=4,
            num_workers=0,
        )

        iter_augmented_loader = iter(augmented_loader)
        aug_wav, aug_genre = next(iter_augmented_loader)
        logger.info(f"Augmented training data shape: {aug_wav.shape}")
        logger.info(f"Augmented training targets: {aug_genre}")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    app()
