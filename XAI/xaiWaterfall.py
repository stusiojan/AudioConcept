import shap
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from AudioConcept.models.classifier_svm import SVMClassifier 
from AudioConcept.config import MODELS_DIR, PROCESSED_DATA_DIR

class xaiWaterfall:
    """
    Klasa xaiWaterfall służy do przetwarzania danych wejściowych, 
    ładowania modelu SVM, skalowania danych, obliczania wartości SHAP oraz generowania wykresu waterfall.
    """
    
    def __init__(self, 
                 features_path: Path = PROCESSED_DATA_DIR / "processed_dataset.csv",
                 model_path: Path = MODELS_DIR / "svm_genre_classifier.pkl",
                 predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
                 sample_size: int = 100,
                 shap_sample_size: int = 20,
                 sample_index: int = 20):
        self.features_path = features_path
        self.model_path = model_path
        self.predictions_path = predictions_path
        self.sample_size = sample_size
        self.shap_sample_size = shap_sample_size
        self.sample_index = sample_index
        
        self.X_test_df = None
        self.X_test_scaled = None
        self.classifier = None
        self.explainer = None
        self.shap_values = None

    def load_data(self):
        df = pd.read_csv(self.features_path).sample(n=self.sample_size, random_state=42).reset_index(drop=True)
        self.X_test_df = df.drop(columns=["Y"])
        return self.X_test_df

    def load_model(self):
        self.classifier = SVMClassifier()
        self.classifier.load_model(self.model_path)
        return self.classifier

    def scale_data(self):
        if self.X_test_df is None:
            raise ValueError("Dane testowe nie zostały wczytane. Uruchom metodę load_data().")
        self.X_test_scaled = self.classifier.scaler.transform(self.X_test_df)
        return self.X_test_scaled

    def compute_shap_values(self):
        if self.X_test_scaled is None:
            raise ValueError("Dane nie zostały przeskalowane. Uruchom metodę scale_data().")
        # Losowa próbka do inicjalizacji explainer'a
        X20 = shap.utils.sample(self.X_test_scaled, self.shap_sample_size)
        self.explainer = shap.Explainer(self.classifier.model.predict, X20)
        self.shap_values = self.explainer(self.X_test_df)
        return self.shap_values

    def plot_waterfall(self, max_display: int = 14):
        if self.shap_values is None:
            raise ValueError("Wartości SHAP nie zostały obliczone. Uruchom metodę compute_shap_values().")
        plt.clf()
        shap.plots.waterfall(self.shap_values[self.sample_index], max_display=max_display)
        fig = plt.gcf()
        return fig
    
    def plot_top_features(self, sample_index: int = 0, top_n: int = 10):
        if self.shap_values is None:
            raise ValueError("Wartości SHAP nie zostały obliczone. Uruchom metodę compute_shap_values().")
        mean_abs_shap = np.abs(self.shap_values.values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        shap_subset = self.shap_values[sample_index, top_indices]
        plt.clf()
        shap.plots.waterfall(shap_subset, max_display=top_n)
        fig = plt.gcf()
        return fig


    def run(self):
        self.load_data()
        self.load_model()
        self.scale_data()
        self.compute_shap_values()
        # self.plot_waterfall()
        self.plot_top_features()

if __name__ == "__main__":
    xai = xaiWaterfall()
    xai.run()
