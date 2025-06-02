import shap
import pandas as pd
import numpy as np
from pathlib import Path

from AudioConcept.models.classifier_svm import SVMClassifier
from AudioConcept.config import MODELS_DIR, PROCESSED_DATA_DIR


class XaiBeeswarm:
    """
    The xaiBeeswarm class is used for processing input data,
    loading the SVM model, scaling the data, computing SHAP values, 
    and generating the beeswarm plot.
    """

    def __init__(
        self,
        features_path: Path = PROCESSED_DATA_DIR / "processed_dataset.csv",
        model_path: Path = MODELS_DIR / "best_SVM_model.pkl",
        sample_size: int = 100,
        shap_sample_size: int = 20,
    ):
        self.features_path = features_path
        self.model_path = model_path
        self.sample_size = sample_size
        self.shap_sample_size = shap_sample_size

        self.X_test_df = None
        self.X_test_scaled = None
        self.classifier = None
        self.explainer = None
        self.shap_values = None

    def load_data(self):
        df = (
            pd.read_csv(self.features_path)
            .sample(n=self.sample_size, random_state=42)
            .reset_index(drop=True)
        )
        self.X_test_df = df.drop(columns=["Y"])
        return self.X_test_df

    def load_model(self):
        self.classifier = SVMClassifier()
        self.classifier.load_model(self.model_path)
        return self.classifier

    def scale_data(self):
        if self.X_test_df is None:
            raise ValueError(
                "Dane testowe nie zostały wczytane. Uruchom metodę load_data()."
            )
        self.X_test_scaled = self.classifier.scaler.transform(self.X_test_df.values)
        return self.X_test_scaled

    def compute_shap_values(self):
        if self.X_test_scaled is None:
            raise ValueError(
                "Dane nie zostały przeskalowane. Uruchom metodę scale_data()."
            )
        X_bg = shap.utils.sample(self.X_test_scaled, self.shap_sample_size)
        self.explainer = shap.Explainer(self.classifier.model.predict, X_bg)
        self.shap_values = self.explainer(self.X_test_df)
        return self.shap_values

    def plot_beeswarm(self, max_display: int = 10):
        if self.shap_values is None:
            raise ValueError(
                "Wartości SHAP nie zostały obliczone. Uruchom metodę compute_shap_values()."
            )
        shap.plots.beeswarm(self.shap_values, max_display=max_display)

    def plot_top_features(self, top_n: int = 10):
        if self.shap_values is None:
            raise ValueError(
                "Wartości SHAP nie zostały obliczone. Uruchom metodę compute_shap_values()."
            )
        mean_abs_shap = np.abs(self.shap_values.values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        shap_subset = self.shap_values[:, top_indices]
        shap.plots.beeswarm(shap_subset)

    def run(self):
        self.load_data()
        self.load_model()
        self.scale_data()
        self.compute_shap_values()
        self.plot_top_features(top_n=10)


if __name__ == "__main__":
    xai = XaiBeeswarm()
    xai.run()
