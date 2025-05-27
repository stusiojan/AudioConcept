import shap
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
from AudioConcept.modeling.svm_classifier import SVMClassifier 
from AudioConcept.config import MODELS_DIR, PROCESSED_DATA_DIR

class xaiWaterfall:
    """
    Klasa xaiWaterfall służy do przetwarzania danych wejściowych, 
    ładowania modelu SVM, skalowania danych, obliczania wartości SHAP oraz generowania wykresu waterfall.
    """
    
    def __init__(self, 
                 features_path: Path = PROCESSED_DATA_DIR / "processed_dataset.csv", # poprawić ścieżkę na pojedynczy wiersz
                 model_path: Path = MODELS_DIR / "svm_genre_classifier.pkl",
                 predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
                 sample_size: int = 50,
                 shap_sample_size: int = 20,
                 sample_index: int = 20):
        self.features_path = features_path
        self.model_path = model_path
        self.predictions_path = predictions_path
        self.sample_size = sample_size
        self.shap_sample_size = shap_sample_size
        self.sample_index = sample_index
        
        # Inicjalizacja atrybutów
        self.X_test_df = None
        self.X_test_scaled = None
        self.classifier = None
        self.explainer = None
        self.shap_values = None

    def load_data(self):
        """Wczytuje dane testowe z pliku CSV, pobiera losową próbkę oraz usuwa kolumnę 'Y'."""
        df = pd.read_csv(self.features_path).sample(n=self.sample_size, random_state=42).reset_index(drop=True)
        self.X_test_df = df.drop(columns=["Y"])
        return self.X_test_df

    def load_model(self):
        """Ładuje model SVM wraz ze skalatorem."""
        self.classifier = SVMClassifier()
        self.classifier.load_model(self.model_path)
        return self.classifier

    def scale_data(self):
        """Skaluje dane testowe przy pomocy skalatora z załadowanego modelu."""
        if self.X_test_df is None:
            raise ValueError("Dane testowe nie zostały wczytane. Uruchom metodę load_data().")
        self.X_test_scaled = self.classifier.scaler.transform(self.X_test_df)
        return self.X_test_scaled

    def compute_shap_values(self):
        """Oblicza wartości SHAP dla danych testowych."""
        if self.X_test_scaled is None:
            raise ValueError("Dane nie zostały przeskalowane. Uruchom metodę scale_data().")
        # Losowa próbka do inicjalizacji explainer'a
        X20 = shap.utils.sample(self.X_test_scaled, self.shap_sample_size)
        self.explainer = shap.Explainer(self.classifier.model.predict, X20)
        self.shap_values = self.explainer(self.X_test_df)
        return self.shap_values

    def plot_waterfall(self, max_display: int = 14):
        """Generuje wykres waterfall dla danego indeksu próbki i wyświetla go w Streamlit."""
        if self.shap_values is None:
            raise ValueError("Wartości SHAP nie zostały obliczone. Uruchom metodę compute_shap_values().")

        plt.clf()
        shap.plots.waterfall(self.shap_values[self.sample_index], max_display=max_display)
        fig = plt.gcf()
        return fig

    def run(self):
        """
        Uruchamia pełen pipeline:
          1. Wczytanie danych testowych.
          2. Załadowanie modelu.
          3. Skalowanie danych.
          4. Obliczenie wartości SHAP.
          5. Generowanie wykresu waterfall.
        """
        self.load_data()
        self.load_model()
        self.scale_data()
        self.compute_shap_values()
        self.plot_waterfall()

# if __name__ == "__main__":
#     xai = xaiWaterfall()
#     xai.run()
