import shap
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from AudioConcept.models.classifier_svm import SVMClassifier
from AudioConcept.config import MODELS_DIR, PROCESSED_DATA_DIR


class XaiWaterfall:
    """
    The xaiWaterfall class is used for processing input data,
    loading the SVM model, scaling the data, computing SHAP values, 
    and generating the waterfall plot.
    """

    def __init__(
        self,
        features_path: Path = PROCESSED_DATA_DIR / "processed_dataset.csv",  # features
        model_path: Path = MODELS_DIR / "best_SVM_model.pkl",
        predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
        sample_size: int = 100,
        shap_sample_size: int = 20,
        sample_index: int = 20,
    ):
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
                "Test data has not been loaded. Run the load_data() method."
            )
        try:
            self.X_test_scaled = self.classifier.scaler.transform(self.X_test_df)
        except ValueError as e:
            st.error(f"Feature mismatch error: {e}")
        return self.X_test_scaled

    def compute_shap_values(self):
        if self.X_test_scaled is None:
            raise ValueError(
                "Data has not been scaled. Run the scale_data() method."
            )
        X20 = shap.utils.sample(self.X_test_scaled, self.shap_sample_size)

        try:
            self.explainer = shap.Explainer(self.classifier.model.predict, X20)
            self.shap_values = self.explainer(self.X_test_scaled)
            self.shap_values.feature_names = list(self.X_test_df.columns)
        except Exception as e:
            st.error(f"Error computing SHAP values: {e}")
            self.explainer = shap.KernelExplainer(
                self.classifier.model.predict_proba, X20
            )
            self.shap_values = self.explainer.shap_values(self.X_test_scaled[0:1])[0]
        return self.shap_values

    def plot_waterfall(self, max_display: int = 14):
        if self.shap_values is None:
            raise ValueError(
                "SHAP values have not been computed. Run the compute_shap_values() method."
            )
        plt.clf()
        shap.plots.waterfall(
            self.shap_values[self.sample_index], max_display=max_display
        )
        fig = plt.gcf()
        return fig

    def plot_top_features(self, sample_index: int = 0, top_n: int = 10):
        if self.shap_values is None:
            raise ValueError(
                "SHAP values have not been computed. Please run the compute_shap_values() method."
            )
        mean_abs_shap = np.abs(self.shap_values.values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        shap_subset = self.shap_values[sample_index, top_indices]

        if (
            hasattr(self.shap_values, "feature_names")
            and self.shap_values.feature_names is not None
        ):
            shap_subset.feature_names = [
                self.shap_values.feature_names[i] for i in top_indices
            ]
        plt.clf()
        shap.plots.waterfall(shap_subset, max_display=top_n)
        fig = plt.gcf()
        return fig

    def run(self):
        self.load_data()
        self.load_model()
        self.scale_data()
        self.compute_shap_values()
        self.plot_top_features()


if __name__ == "__main__":
    xai = XaiWaterfall()
    xai.run()
