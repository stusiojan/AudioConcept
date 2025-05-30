import os
import random
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from torch.utils import data
from typing import Tuple, Optional
import warnings
from config import DATA_PATH, GTZAN_GENRES

warnings.filterwarnings("ignore")


class SpectrogramTransforms:
    """Spectrogram-based augmentations - more efficient and effective"""

    @staticmethod
    def time_mask(spec: np.ndarray, max_mask_pct: float = 0.1) -> np.ndarray:
        """Mask random time steps"""
        n_time = spec.shape[1]
        mask_size = int(n_time * max_mask_pct * random.random())
        if mask_size > 0:
            mask_start = random.randint(0, n_time - mask_size)
            spec[:, mask_start : mask_start + mask_size] = 0
        return spec

    @staticmethod
    def freq_mask(spec: np.ndarray, max_mask_pct: float = 0.15) -> np.ndarray:
        """Mask random frequency bins"""
        n_freq = spec.shape[0]
        mask_size = int(n_freq * max_mask_pct * random.random())
        if mask_size > 0:
            mask_start = random.randint(0, n_freq - mask_size)
            spec[mask_start : mask_start + mask_size, :] = 0
        return spec

    @staticmethod
    def mixup_spec(
        spec1: np.ndarray, spec2: np.ndarray, alpha: float = 0.2
    ) -> np.ndarray:
        """Simple spectral mixing"""
        lam = np.random.beta(alpha, alpha)
        return lam * spec1 + (1 - lam) * spec2


class AudioProcessor:
    """Handles audio loading and spectrogram conversion with optimized performance"""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        duration: float = 29.12,  # Full GTZAN track length (original 30s files minus tiny buffer)
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.target_length = int(sample_rate * duration)

        # Track cache stats for debugging
        self.cache_hits = 0
        self.cache_misses = 0

        # Simple in-memory cache (limited to avoid memory issues)
        self._mel_cache = {}
        self._max_cache_size = 200  # Limit cache size

    def load_audio(self, file_path: str) -> np.ndarray:
        """Load and preprocess audio file with efficient loading"""
        try:
            # Fast loading with soundfile if possible
            try:
                # Use already imported soundfile
                audio, sr = sf.read(file_path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)  # Convert to mono if needed

                # Resample if necessary (less common with standard datasets)
                if sr != self.sample_rate:
                    audio = librosa.resample(
                        audio, orig_sr=sr, target_sr=self.sample_rate
                    )
            except Exception as sf_error:
                # Fallback to librosa
                print(f"Soundfile error, using librosa fallback: {sf_error}")
                audio, sr = librosa.load(
                    file_path, sr=self.sample_rate, mono=True, res_type="kaiser_fast"
                )

            # Handle length - crop or pad to target length
            if len(audio) > self.target_length:
                # Center crop for consistent features
                start = (len(audio) - self.target_length) // 2
                audio = audio[start : start + self.target_length]
            else:
                # Pad with zeros if too short
                pad_length = self.target_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode="constant")

            return audio.astype(np.float32)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.zeros(self.target_length, dtype=np.float32)

    def audio_to_melspec(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to mel-spectrogram with efficient processing"""
        # Use cache based on audio hash if enabled
        audio_hash = hash(audio.tobytes())
        if audio_hash in self._mel_cache:
            self.cache_hits += 1
            return self._mel_cache[audio_hash]

        self.cache_misses += 1

        # Compute mel-spectrogram with optimized parameters
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmax=8000,  # Limit to 8kHz for music
            power=2.0,  # Use power spectrogram
        )

        # Convert to log scale (dB) - improved normalization
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

        # Normalize to [0, 1] range for better gradient flow
        mel_spec_db = (mel_spec_db + 80) / 80
        mel_spec_db = np.clip(mel_spec_db, 0, 1)

        result = mel_spec_db.astype(np.float32)

        # Update cache (with size limit)
        if len(self._mel_cache) < self._max_cache_size:
            self._mel_cache[audio_hash] = result

        return result


class ImprovedGTZANDataset(data.Dataset):
    """Improved GTZAN Dataset with proper spectrogram processing"""

    def __init__(
        self,
        data_path: str,
        split: str,
        genres: list,
        sample_rate: int = 22050,
        n_mels: int = 128,
        augment: bool = False,
        num_chunks: int = 1,
    ):

        self.data_path = data_path
        self.split = split
        self.genres = genres
        self.augment = augment and (split == "train")
        self.num_chunks = num_chunks

        # Initialize audio processor
        self.processor = AudioProcessor(sample_rate=sample_rate, n_mels=n_mels)

        # Load file list
        self._load_file_list()

        # print(f"Loaded {len(self.file_list)} files for {split} split")

    def _load_file_list(self):
        """Load list of audio files"""
        list_file = os.path.join(self.data_path, f"{self.split}_filtered.txt")

        if os.path.exists(list_file):
            # Load from existing split file
            with open(list_file, "r") as f:
                self.file_list = [line.strip() for line in f.readlines()]
        else:
            # Create file list from directory structure
            self.file_list = []
            genres_dir = os.path.join(self.data_path, "genres")

            for genre in self.genres:
                genre_dir = os.path.join(genres_dir, genre)
                if os.path.exists(genre_dir):
                    for file in os.listdir(genre_dir):
                        if file.endswith(".wav"):
                            self.file_list.append(f"{genre}/{file}")

            print(f"Created file list with {len(self.file_list)} files")

    def _apply_spec_augmentation(self, mel_spec: np.ndarray) -> np.ndarray:
        """Apply spectrogram augmentations"""
        if not self.augment:
            return mel_spec

        # Time and frequency masking
        if random.random() < 0.5:
            mel_spec = SpectrogramTransforms.time_mask(mel_spec)

        if random.random() < 0.6:
            mel_spec = SpectrogramTransforms.freq_mask(mel_spec)

        # Random gain (in log domain)
        if random.random() < 0.3:
            gain = random.uniform(-0.1, 0.1)
            mel_spec = np.clip(mel_spec + gain, 0, 1)

        return mel_spec

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        try:
            file_path = self.file_list[index]

            # Extract genre from file path
            genre_name = file_path.split("/")[0]
            genre_idx = self.genres.index(genre_name)

            # Load audio
            full_audio_path = os.path.join(self.data_path, "genres", file_path)
            audio = self.processor.load_audio(full_audio_path)

            # Always use single chunk processing for simplicity and consistency
            # This will help with initial training performance
            mel_spec = self.processor.audio_to_melspec(audio)

            # Apply augmentation only during training
            if self.augment and self.split == "train":
                mel_spec = self._apply_spec_augmentation(mel_spec)

            # Add channel dimension for CNN (C, H, W)
            mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)

            return mel_spec, genre_idx

        except Exception as e:
            print(f"Error processing {index}: {e}")
            # Return dummy data with consistent shape
            return torch.zeros(1, 128, 1292), 0  # Typical mel-spec shape

    def __len__(self) -> int:
        return len(self.file_list)


def get_improved_dataloader(
    split: str,
    genres: list = GTZAN_GENRES,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = False,
    num_chunks: int = 1,
    data_path: str = DATA_PATH,
) -> data.DataLoader:
    """Create improved data loader"""

    dataset = ImprovedGTZANDataset(
        data_path=data_path,
        split=split,
        genres=genres,
        augment=augment,
        num_chunks=num_chunks,
    )

    # Adjust batch size for multi-chunk data
    if split != "train" and num_chunks > 1:
        batch_size = max(1, batch_size // num_chunks)

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=False,  # Disable persistent workers to avoid file descriptor leaks
        prefetch_factor=(
            2 if num_workers > 0 else None
        ),  # Reduce prefetch to lower memory footprint
    )


# Create data loaders
train_loader = get_improved_dataloader(
    split="train",
    batch_size=16,
    augment=True,
)
valid_loader = get_improved_dataloader(
    split="valid",
    batch_size=16,
    augment=False,
)
test_loader = get_improved_dataloader(
    split="test",
    batch_size=16,
    augment=False,
)


# Example usage and testing
def test_improved_pipeline():
    """Test the improved pipeline"""

    # Example genres (adjust to your actual genres)
    GENRES = GTZAN_GENRES

    # Create test dataset (adjust path)
    data_path = DATA_PATH

    try:
        # Create data loaders
        train_loader = get_improved_dataloader(
            data_path=data_path,
            split="train",
            genres=GENRES,
            batch_size=16,
            augment=True,
        )

        # Test loading a batch
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(
                f"  Data shape: {data.shape}"
            )  # Should be [batch_size, 1, n_mels, time]
            print(f"  Target shape: {target.shape}")
            print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
            break

    except Exception as e:
        print(f"Test failed: {e}")
        print("Please update the data_path variable to point to your GTZAN dataset")


if __name__ == "__main__":
    test_improved_pipeline()
