import librosa
from loguru import logger
import numpy as np


"""
Feature extraction for audio files.
Saves extracted features to the processed data directory.
"""


class AudioFeatureExtractor:
    """
    Features extracted:
    - Length
    - Chroma STFT (mean, variance)
    - RMS (mean, variance)
    - Spectral centroid (variance)
    - Spectral bandwidth (variance)
    - Spectral rolloff (mean, variance)
    - Zero crossing rate (mean, variance)
    - Harmony and percussive components (mean, variance)
    - Tempo
    - MFCC coefficients 1-20 (mean, variance)
    """

    def __init__(self, n_mfcc=20, hop_length=512, n_fft=2048):
        """
        Initialize the feature extractor.

        Args:
            n_mfcc (int): Number of MFCC coefficients to extract (default: 20)
            hop_length (int): Number of samples between successive frames (default: 512)
            n_fft (int): Length of the FFT window (default: 2048)
        """
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract_features(self, audio_data, sample_rate):
        """
        Extract audio features.

        Args:
            audio_data (numpy.ndarray): Audio time series
            sample_rate (int): Sample rate of audio data

        Returns:
            dict: Dictionary containing all extracted features
        """
        features = {}

        try:
            features["length"] = len(audio_data) / sample_rate

            chroma_stft = librosa.feature.chroma_stft(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )
            features["chroma_stft_mean"] = np.mean(chroma_stft)
            features["chroma_stft_var"] = np.var(chroma_stft)

            rms = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)[0]
            features["rms_mean"] = np.mean(rms)
            features["rms_var"] = np.var(rms)

            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )[0]
            features["spectral_centroid_var"] = np.var(spectral_centroid)

            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )[0]
            features["spectral_bandwidth_var"] = np.var(spectral_bandwidth)

            rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )[0]
            features["rolloff_mean"] = np.mean(rolloff)
            features["rolloff_var"] = np.var(rolloff)

            zcr = librosa.feature.zero_crossing_rate(
                audio_data, hop_length=self.hop_length
            )[0]
            features["zero_crossing_rate_mean"] = np.mean(zcr)
            features["zero_crossing_rate_var"] = np.var(zcr)

            harmonic, percussive = librosa.effects.hpss(audio_data)
            features["harmony_mean"] = np.mean(harmonic)
            features["harmony_var"] = np.var(harmonic)
            features["perceptr_mean"] = np.mean(percussive)
            features["perceptr_var"] = np.var(percussive)

            tempo, _ = librosa.beat.beat_track(
                y=audio_data, sr=sample_rate, hop_length=self.hop_length
            )
            features["tempo"] = float(tempo)

            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
            )

            for i in range(1, self.n_mfcc + 1):
                mfcc_values = mfccs[i - 1]
                features[f"mfcc{i}_mean"] = np.mean(mfcc_values)
                features[f"mfcc{i}_var"] = np.var(mfcc_values)

            logger.info(f"Successfully extracted {len(features)} features")

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise

        return features
