from pathlib import Path

import librosa
from loguru import logger
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import typer

from AudioConcept.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()

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
            features["length"] = len(audio_data)

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

            logger.debug(f"Successfully extracted {len(features)} features")

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise

        return features

    def scale_features(
        self,
        input_path: Path = PROCESSED_DATA_DIR / "processed_audio.csv",
        output_path: Path = PROCESSED_DATA_DIR / "processed_audio.csv",
    ):
        """
        Scale features and save the processed dataset.
        Args:
            input_path (Path): Path to the input CSV file containing audio features
            output_path (Path): Path to save the processed CSV file

        Returns:
            pd.DataFrame: Processed DataFrame with standardized features and encoded labels
        """
        logger.info(f"Processing dataset from {input_path}...")
        df = pd.read_csv(input_path, sep=",")
        logger.info(f"Dataframe1: {df}")

        df_filtered = df.iloc[:, 1:-1]
        y = df.iloc[:, -1]

        logger.info("Calculating correlation matrix...")
        corr_matrix = df_filtered.corr()

        logger.info("Removing highly correlated features...")
        correlated_features = set()

        for i in tqdm(
            range(len(corr_matrix.columns)), desc="Identifying correlated features"
        ):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    colname = corr_matrix.columns[i]
                    if colname not in correlated_features:
                        if np.var(df_filtered[colname]) < np.var(
                            df_filtered[corr_matrix.columns[j]]
                        ):
                            correlated_features.add(corr_matrix.columns[i])
                        else:
                            correlated_features.add(corr_matrix.columns[j])

        df = df.drop(columns=correlated_features)
        # df = df.drop(columns={"filename", "label"})

        logger.info(f"Features: {df.columns.tolist()}")
        logger.info(f"Features values: {df.values}")

        logger.info("Standardizing features...")
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(df)

        logger.info("Encoding labels...")
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        X_standardized_df = pd.DataFrame(X_standardized, columns=df.columns)
        logger.info(f"Features: {X_standardized_df.columns.tolist()}")
        logger.info(f"Features values: {X_standardized_df.values}")

        df_final = X_standardized_df
        # df_final = pd.concat(
        #     [X_standardized_df, pd.Series(y_encoded, name="Y")], axis=1
        # )

        logger.info(
            f"Number of features (columns) in final dataset: {df_final.shape[1]}"
        )

        logger.info(f"Saving processed dataset to {output_path}...")
        df_final.to_csv(output_path, index=False)

        logger.success(f"Processing complete. Saved to {output_path}.")

        return df_final


def process_gtzan_features(
    input_path: Path = RAW_DATA_DIR / "features_30_sec.csv",
    output_path: Path = PROCESSED_DATA_DIR / "processed_features.csv",
):
    """
    Process and scale the GTZAN dataset features.
    """
    logger.info(f"Processing dataset from {input_path}...")
    df = pd.read_csv(input_path, sep=";")

    df_filtered = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    logger.info("Calculating correlation matrix...")
    corr_matrix = df_filtered.corr()

    logger.info("Removing highly correlated features...")
    correlated_features = set()

    for i in tqdm(
        range(len(corr_matrix.columns)), desc="Identifying correlated features"
    ):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                colname = corr_matrix.columns[i]
                if colname not in correlated_features:
                    if np.var(df_filtered[colname]) < np.var(
                        df_filtered[corr_matrix.columns[j]]
                    ):
                        correlated_features.add(corr_matrix.columns[i])
                    else:
                        correlated_features.add(corr_matrix.columns[j])

    df = df.drop(columns=correlated_features)
    df = df.drop(columns={"filename", "label"})

    logger.info(f"Number of features (columns) in final dataset: {df.shape[1]}")

    logger.info("Standardizing features...")
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(df)

    scaler_path = PROCESSED_DATA_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved StandardScaler to {scaler_path}")

    logger.info("Encoding labels...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_standardized_df = pd.DataFrame(X_standardized, columns=df.columns)

    df_final = pd.concat([X_standardized_df, pd.Series(y_encoded, name="Y")], axis=1)

    logger.info(f"Number of features (columns) in final dataset: {df_final.shape[1]}")

    logger.info(f"Saving processed dataset to {output_path}...")
    df_final.to_csv(output_path, index=False)

    logger.success(f"Processing complete. Saved to {output_path}.")


@app.command()
def main():
    process_gtzan_features()


if __name__ == "__main__":
    app()
