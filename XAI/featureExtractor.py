import pandas as pd
from pathlib import Path
import librosa
import numpy as np
from AudioConcept.config import PROCESSED_DATA_DIR


class FeatureExtractor:
    def __init__(self, output_path: Path = PROCESSED_DATA_DIR / "test_features.csv"):
        self.output_path = Path(output_path)

    def extract_features(self, file_path: str) -> pd.DataFrame:
        y, sr = librosa.load(file_path, mono=True)
        features = {}

        # Podstawowe cechy
        features['length'] = librosa.get_duration(y=y, sr=sr)

        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = np.mean(chroma_stft)
        features['chroma_stft_var'] = np.var(chroma_stft)

        # RMS
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)

        # Spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_var'] = np.var(spec_cent)

        # Spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_var'] = np.var(spec_bw)

        # Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_var'] = np.var(rolloff)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_var'] = np.var(zcr)

        # Harmony & Perceptual spread
        y_harm, y_perc = librosa.effects.hpss(y)
        features['harmony_mean'] = np.mean(y_harm)
        features['harmony_var'] = np.var(y_harm)
        features['perceptr_mean'] = np.mean(y_perc)
        features['perceptr_var'] = np.var(y_perc)

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            features[f'mfcc{i}_mean'] = np.mean(mfcc[i-1])
            features[f'mfcc{i}_var'] = np.var(mfcc[i-1])

        return pd.DataFrame([features])

    def extract_and_save(self, file_path: str):
        df = self.extract_features(file_path)
        df.to_csv(self.output_path, index=False)
        print(f"Features saved to: {self.output_path}")

# if __name__ == "__main__":
    # extractor = FeatureExtractor()
    # extractor.extract_and_save("C:/Users/weral/Desktop/WIMU/projekt/dataset/genres_original/jazz/jazz.00000.wav")
