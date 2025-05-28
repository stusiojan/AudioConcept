import numpy as np
import librosa
import librosa.effects
import random
from typing import List, Callable, Optional
import scipy.signal
import gc  # For garbage collection


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
            # Ensure audio remains float32 to control memory usage
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
        return audio


class RandomResizedCrop(AudioTransform):
    """Randomly crop and resize audio to target length"""

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        if len(audio) <= self.n_samples:
            # Pad if too short
            pad_length = self.n_samples - len(audio)
            return np.pad(audio, (0, pad_length), mode="constant")

        # Random crop
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

        # Calculate signal power in dB (with numerical stability)
        signal_power = np.mean(audio**2) + 1e-10
        signal_power_db = 10 * np.log10(signal_power)

        # Calculate required noise power
        noise_power_db = signal_power_db - snr_db
        noise_power_linear = 10 ** (noise_power_db / 10)

        # Generate and scale noise
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
                # High-pass filter: remove low frequencies
                cutoff_freq = random.uniform(50, 1000)  # Hz
                sos = scipy.signal.butter(
                    4, cutoff_freq, btype="high", fs=sr, output="sos"
                )
            else:
                # Low-pass filter: remove high frequencies
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
            # Random delay between 50ms and 200ms (reduced for memory)
            delay_ms = random.uniform(50, 200)
            delay_samples = int(delay_ms * sr / 1000)

            if delay_samples >= len(audio) or delay_samples <= 0:
                return audio

            # Random feedback and mix levels (reduced for stability)
            feedback = random.uniform(0.1, 0.4)
            mix = random.uniform(0.2, 0.5)

            # Create output array
            output = audio.copy().astype(np.float32)

            # Apply delay with feedback (more efficient loop)
            for i in range(delay_samples, len(audio)):
                output[i] += output[i - delay_samples] * feedback

            return (audio * (1 - mix) + output * mix).astype(np.float32)

        except Exception as e:
            print(f"Delay error: {e}")
            return audio


class PitchShift(AudioTransform):
    """Memory-optimized pitch shifting"""

    def __init__(self, n_samples: int, sample_rate: int = 22050):
        self.n_samples = n_samples
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        try:
            # Smaller pitch shift range for stability
            n_steps = random.uniform(-2, 2)

            # Use librosa's pitch shifting with optimized parameters
            shifted = librosa.effects.pitch_shift(
                audio,
                sr=sr,
                n_steps=n_steps,
                bins_per_octave=12,  # Reduced for memory efficiency
                res_type="kaiser_fast",  # Faster resampling
            )

            # Ensure correct length
            if len(shifted) != self.n_samples:
                if len(shifted) < self.n_samples:
                    pad_length = self.n_samples - len(shifted)
                    shifted = np.pad(shifted, (0, pad_length), mode="constant")
                else:
                    shifted = shifted[: self.n_samples]

            return shifted.astype(np.float32)

        except Exception as e:
            print(f"Pitch shift error: {e}")
            return audio


class TimeStretch(AudioTransform):
    """Memory-optimized time stretching"""

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        try:
            # Smaller stretch range for stability
            rate = random.uniform(0.9, 1.1)

            # Use librosa with optimized parameters
            stretched = librosa.effects.time_stretch(
                audio,
                rate=rate,
                hop_length=512,  # Larger hop length for efficiency
                n_fft=1024,  # Smaller FFT size for memory
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


class Reverb(AudioTransform):
    """Lightweight reverb implementation"""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        try:
            # Shorter reverb time to reduce memory usage
            reverb_time = random.uniform(0.3, 1.0)  # seconds
            decay_rate = random.uniform(0.3, 0.7)

            # Generate smaller impulse response
            ir_length = int(reverb_time * sr)
            ir_length = min(ir_length, len(audio) // 4)  # Limit IR length

            if ir_length <= 0:
                return audio

            ir = np.random.normal(0, 1, ir_length).astype(np.float32)

            # Apply exponential decay
            decay = np.exp(-np.arange(ir_length) * decay_rate / sr)
            ir = ir * decay

            # Use scipy's fftconvolve for efficiency with mode='same'
            reverb_signal = scipy.signal.fftconvolve(audio, ir, mode="same")
            reverb_signal = reverb_signal[: len(audio)].astype(np.float32)

            # Mix with dry signal
            mix_level = random.uniform(0.15, 0.35)
            result = audio * (1 - mix_level) + reverb_signal * mix_level

            return result.astype(np.float32)

        except Exception as e:
            print(f"Reverb error: {e}")
            return audio


class SpectralRolloff(AudioTransform):
    """Memory-optimized spectral rolloff"""

    def __init__(self):
        pass

    def __call__(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        try:
            # Use smaller FFT parameters for memory efficiency
            n_fft = min(1024, len(audio) // 4)
            hop_length = n_fft // 4

            if n_fft <= 0:
                return audio

            # Compute STFT with smaller parameters
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Random spectral modification
            rolloff_factor = random.uniform(0.8, 1.2)

            # Apply frequency-dependent gain to higher frequencies only
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            for i, freq in enumerate(freqs):
                if freq > 1000:  # Only affect higher frequencies
                    gain = rolloff_factor ** ((freq - 1000) / 8000)
                    magnitude[i] *= gain

            # Reconstruct audio
            modified_stft = magnitude * np.exp(1j * phase)
            result = librosa.istft(
                modified_stft, hop_length=hop_length, length=len(audio)
            )

            return result.astype(np.float32)

        except Exception as e:
            print(f"Spectral rolloff error: {e}")
            return audio
