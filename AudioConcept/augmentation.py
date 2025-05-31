import random
from typing import List

import librosa
import librosa.effects
import numpy as np
import scipy.signal

"""
Due to dependency issues with torchaudio and librosa, we implement audio transformations.

Due to memory constraints, we use numpy arrays instead of torch tensors.
"""


class AudioTransform:
    """Base class for audio transformations"""

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        raise NotImplementedError


class RandomApply:
    """Randomly apply a transformation with given probability"""

    def __init__(self, transforms: List[AudioTransform], p: float = 0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        if random.random() < self.p:
            transform = random.choice(self.transforms)
            return transform(audio, sr)
        return audio


class Compose:
    """Compose multiple transformations"""

    def __init__(self, transforms: List[AudioTransform]):
        self.transforms = transforms

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        for transform in self.transforms:
            audio = transform(audio, sr)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
        return audio


class RandomResizedCrop(AudioTransform):
    """Randomly crop and resize audio to target length"""

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        if len(audio) <= self.n_samples:
            pad_length = self.n_samples - len(audio)
            return np.pad(audio, (0, pad_length), mode="constant")

        start_idx = random.randint(0, len(audio) - self.n_samples)
        return audio[start_idx : start_idx + self.n_samples]


class PolarityInversion(AudioTransform):
    """Invert the polarity of the audio signal"""

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        return -audio


class Noise(AudioTransform):
    """Add random noise to audio with better SNR control"""

    def __init__(self, min_snr_db: float = 10, max_snr_db: float = 30):
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        snr_db = random.uniform(self.min_snr_db, self.max_snr_db)

        signal_power = np.mean(audio**2) + 1e-10
        signal_power_db = 10 * np.log10(signal_power)

        noise_power_db = signal_power_db - snr_db
        noise_power_linear = 10 ** (noise_power_db / 10)

        noise = np.random.normal(0, np.sqrt(noise_power_linear), audio.shape).astype(
            np.float32
        )

        return (audio + noise).astype(np.float32)


class Gain(AudioTransform):
    """Apply random gain to audio with dB control"""

    def __init__(self, min_gain_db: float = -6, max_gain_db: float = 6):
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        gain_linear = 10 ** (gain_db / 20)
        return (audio * gain_linear).astype(np.float32)


class HighLowPass(AudioTransform):
    """Apply high-pass or low-pass filter using scipy (more memory efficient)"""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        try:
            filter_type = random.choice(["high", "low"])

            if filter_type == "high":
                cutoff_freq = random.uniform(50, 1000)  # Hz
                sos = scipy.signal.butter(
                    4, cutoff_freq, btype="high", fs=sr, output="sos"
                )
            else:
                cutoff_freq = random.uniform(2000, 8000)  # Hz
                sos = scipy.signal.butter(
                    4, cutoff_freq, btype="low", fs=sr, output="sos"
                )

            filtered = scipy.signal.sosfilt(sos, audio)
            return filtered.astype(np.float32)

        except Exception as e:
            print(f"Filter error: {e}")
            return audio


class Delay(AudioTransform):
    """Optimized delay/echo effect"""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        try:
            delay_ms = random.uniform(50, 200)
            delay_samples = int(delay_ms * sr / 1000)

            if delay_samples >= len(audio) or delay_samples <= 0:
                return audio

            feedback = random.uniform(0.1, 0.4)
            mix = random.uniform(0.2, 0.5)

            output = audio.copy().astype(np.float32)

            for i in range(delay_samples, len(audio)):
                output[i] += output[i - delay_samples] * feedback

            return (audio * (1 - mix) + output * mix).astype(np.float32)

        except Exception as e:
            print(f"Delay error: {e}")
            return audio


class TimeStretch(AudioTransform):
    """Memory-optimized time stretching"""

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        try:
            rate = random.uniform(0.9, 1.1)

            stretched = librosa.effects.time_stretch(
                audio,
                rate=rate,
                hop_length=512,
                n_fft=1024,
            )

            # Resize to target length
            if len(stretched) != self.n_samples:
                if len(stretched) < self.n_samples:
                    pad_length = self.n_samples - len(stretched)
                    stretched = np.pad(stretched, (0, pad_length), mode="constant")
                else:
                    stretched = stretched[: self.n_samples]

            return stretched.astype(np.float32)

        except Exception as e:
            print(f"Time stretch error: {e}")
            return audio
