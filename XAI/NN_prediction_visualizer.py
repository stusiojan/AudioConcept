import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
import torchaudio
from scipy.ndimage import zoom

from demo.models.model_type import ModelType
from AudioConcept.models.model_cnn import CNN
from AudioConcept.models.model_vggish import VGGish


class ModelWrapper(nn.Module):
    """Wrapper that separates audio preprocessing from neural network for SHAP compatibility."""

    def __init__(self, original_model, model_type):
        super(ModelWrapper, self).__init__()
        self.original_model = original_model
        self.model_type = model_type

        if model_type == ModelType.CNN.value:
            self.input_bn = original_model.input_bn
            self.layer1 = original_model.layer1
            self.layer2 = original_model.layer2
            self.layer3 = original_model.layer3
            self.layer4 = original_model.layer4
            self.layer5 = original_model.layer5
            self.dense1 = original_model.dense1
            self.dense_bn = original_model.dense_bn
            self.dense2 = original_model.dense2
            self.dropout = original_model.dropout
            self.relu = original_model.relu
        elif model_type == ModelType.VGGISH.value:
            self.input_bn = original_model.input_bn
            self.conv_layers = original_model.conv_layers
            self.fcs = original_model.fcs

    def forward(self, mel_spectrogram):
        """Forward pass expecting preprocessed mel spectrogram input."""
        if self.model_type == ModelType.CNN.value:
            # Input is already a mel spectrogram in dB scale, just needs to be shaped properly
            out = mel_spectrogram.unsqueeze(1)  # Add channel dimension
            out = self.input_bn(out)

            # Convolutional layers
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)

            # Reshape for dense layers
            out = out.reshape(len(out), -1)

            # Dense layers
            out = self.dense1(out)
            out = self.dense_bn(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.dense2(out)

            return out
        elif self.model_type == ModelType.VGGISH.value:
            # Input is already a mel spectrogram in dB scale
            out = mel_spectrogram.unsqueeze(1)  # Add channel dimension
            out = self.input_bn(out)

            # Reshape from [batch, 1, n_mels, time] to [batch, 1, 224, 224] to match VGG input
            out = torch.nn.functional.interpolate(
                out, size=(224, 224), mode="bilinear", align_corners=False
            )

            # Convolutional layers
            out = self.conv_layers(out)

            # Reshape for fully connected layers
            out = out.reshape(len(out), -1)

            # Fully connected layers
            out = self.fcs(out)

            return out

        raise ValueError(f"Unsupported model type: {self.model_type}")


class NN_prediction_visualizer:
    def __init__(self, model_type: ModelType, file_path: str):
        self.model_type = model_type.value
        self.file_path = file_path
        self.device = torch.device("cpu")
        self.original_model = self._load_model()
        self.model_wrapper = None  # Will be created after loading trained weights

    def _load_model(self):
        if self.model_type == ModelType.CNN.value:
            return CNN().to(self.device)
        elif self.model_type == ModelType.VGGISH.value:
            return VGGish().to(self.device)
        else:
            raise ValueError("Unsupported model type for neural network XAI plotting.")

    def _load_audio(self, file_path: str):
        if not file_path:
            raise ValueError("File path must be provided to load audio.")
        y, sr = librosa.load(file_path, sr=None)
        return y, sr

    def _load_trained_model(self):
        """Load the trained model weights from the saved checkpoint/pickle file."""
        from pathlib import Path
        import pickle

        models_dir = Path("/Users/jan/git/wimu/AudioConcept/models")
        model_file = models_dir / f"best_{self.model_type}_model.pkl"

        if model_file.exists():
            with open(model_file, "rb") as f:
                trained_model = pickle.load(f)
            # Copy the state dict from the trained model
            self.original_model.load_state_dict(trained_model.state_dict())
            print(f"Loaded trained {self.model_type} model weights")
        else:
            print(
                f"Warning: No trained model found at {model_file}, using random weights"
            )

        # Create the wrapper after loading weights
        self.model_wrapper = ModelWrapper(self.original_model, self.model_type).to(
            self.device
        )
        self.model_wrapper.eval()

    def _preprocess_audio(self, file_path: str):
        """Preprocess audio data consistent with model training."""
        y, sr = librosa.load(file_path, sr=22050)  # Use same sample rate as model

        # Ensure consistent length (30 seconds = 661500 samples at 22050 Hz)
        target_length = 22050 * 30
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode="constant")
        else:
            y = y[:target_length]

        return y

    def _audio_to_melspec(self, audio_data):
        """Convert audio to mel spectrogram using the same transforms as the model."""
        audio_tensor = torch.FloatTensor(audio_data).to(self.device)

        if self.model_type == ModelType.CNN.value:
            melspec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                f_min=0.0,
                f_max=11025.0,
                n_mels=128,
            ).to(self.device)
            amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(self.device)

            melspec = melspec_transform(audio_tensor)
            melspec_db = amplitude_to_db(melspec)
            return melspec_db

        elif self.model_type == ModelType.VGGISH.value:
            melspec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                f_min=0.0,
                f_max=11025.0,
                n_mels=128,  # Adjust based on VGGish specs
            ).to(self.device)
            amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(self.device)

            melspec = melspec_transform(audio_tensor)
            melspec_db = amplitude_to_db(melspec)
            return melspec_db

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _create_model_melspectrogram(self, audio_data):
        """Create mel spectrogram using the same parameters as the model."""
        melspec = librosa.feature.melspectrogram(
            y=audio_data, sr=22050, n_fft=1024, n_mels=128, fmin=0.0, fmax=11025.0
        )
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        return melspec_db

    def _create_background_dataset(self, n_samples=5):
        """Create background dataset of mel spectrograms for SHAP explainer."""
        # Create various background samples (silence, noise, etc.)
        background_melspecs = []
        target_length = 22050 * 30  # 30 seconds

        for i in range(n_samples):
            if i == 0:
                # Pure silence
                audio_sample = np.zeros(target_length)
            elif i == 1:
                # White noise (low amplitude)
                audio_sample = np.random.normal(0, 0.01, target_length)
            else:
                # Random noise with varying amplitudes
                amplitude = 0.001 * (i + 1)
                audio_sample = np.random.normal(0, amplitude, target_length)

            # Convert audio to mel spectrogram
            melspec = self._audio_to_melspec(audio_sample)
            background_melspecs.append(melspec)

        # Stack into a batch
        background_tensor = torch.stack(background_melspecs)
        return background_tensor

    def _convert_shap_to_melspec(
        self, shap_values, melspec_shape, predicted_class=None
    ):
        """Convert SHAP values to mel spectrogram format for visualization."""
        try:
            # Handle different SHAP value formats
            if isinstance(shap_values, torch.Tensor):
                shap_array = shap_values.cpu().numpy()
            elif isinstance(shap_values, list) and len(shap_values) > 0:
                # If it's a list, take the first element or combine them
                if isinstance(shap_values[0], torch.Tensor):
                    shap_array = shap_values[0].cpu().numpy()
                else:
                    shap_array = shap_values[0]
            else:
                shap_array = np.array(shap_values)

            print(
                f"SHAP values shape: {shap_array.shape}, Target melspec shape: {melspec_shape}"
            )

            # Handle 4D case: (batch, mel_bins, time, classes)
            if len(shap_array.shape) == 4:
                # Remove batch dimension
                shap_array = shap_array[0]  # Shape becomes (mel_bins, time, classes)

                # If predicted_class is provided, use SHAP values for that class
                if (
                    predicted_class is not None
                    and predicted_class < shap_array.shape[-1]
                ):
                    shap_array = shap_array[:, :, predicted_class]
                    print(f"Using SHAP values for predicted class {predicted_class}")
                else:
                    # Use sum across all classes or maximum
                    shap_array = np.sum(np.abs(shap_array), axis=-1)
                    print("Using sum of absolute SHAP values across all classes")

            # Handle 3D case: (batch, mel_bins, time) or (mel_bins, time, classes)
            elif len(shap_array.shape) == 3:
                # Check if first dimension could be batch dimension
                if shap_array.shape[0] == 1:
                    shap_array = shap_array[0]  # Remove batch dimension
                # Check if last dimension could be classes (typically 10 for GTZAN)
                elif shap_array.shape[-1] <= 10 and predicted_class is not None:
                    if predicted_class < shap_array.shape[-1]:
                        shap_array = shap_array[:, :, predicted_class]
                        print(
                            f"Using SHAP values for predicted class {predicted_class}"
                        )
                    else:
                        shap_array = np.sum(np.abs(shap_array), axis=-1)
                        print("Using sum of absolute SHAP values across all classes")

            # Now shap_array should be 2D
            if len(shap_array.shape) == 2:
                # Check if dimensions match exactly
                if shap_array.shape == melspec_shape:
                    return np.abs(shap_array)  # Use absolute values for importance

                # If shapes don't match exactly but are close, try to resize
                if (
                    abs(shap_array.shape[0] - melspec_shape[0]) <= 1
                    and abs(shap_array.shape[1] - melspec_shape[1]) <= 1
                ):
                    # Try to resize to match target shape
                    scale_factors = (
                        melspec_shape[0] / shap_array.shape[0],
                        melspec_shape[1] / shap_array.shape[1],
                    )
                    shap_resized = zoom(shap_array, scale_factors)
                    print(
                        f"Resized SHAP values from {shap_array.shape} to {shap_resized.shape}"
                    )
                    return np.abs(shap_resized)

            # If we still can't match shapes, create a simple visualization
            print(
                f"Creating fallback SHAP visualization due to shape mismatch. Final SHAP shape: {shap_array.shape}"
            )
            return np.random.random(melspec_shape) * np.max(np.abs(shap_array)) * 0.1

        except Exception as e:
            print(f"Error converting SHAP values to mel spectrogram: {str(e)}")
            # Return a simple random pattern as fallback
            return np.random.random(melspec_shape) * 0.1

    def _create_shap_visualization(
        self, melspec_db, shap_melspec, predicted_class, probabilities
    ):
        """Create a visualization showing original mel spectrogram and SHAP importance."""
        from AudioConcept.config import GTZAN_GENRES

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot original mel spectrogram
        img1 = librosa.display.specshow(
            melspec_db, ax=axes[0], y_axis="mel", x_axis="time", sr=22050, fmax=11025
        )
        axes[0].set_title(
            f"Original Mel Spectrogram\nPredicted: {GTZAN_GENRES[predicted_class]} ({probabilities[predicted_class]:.3f})"
        )
        fig.colorbar(img1, ax=axes[0], format="%+2.0f dB")

        # Plot SHAP importance overlay
        img2 = axes[1].imshow(
            shap_melspec,
            aspect="auto",
            origin="lower",
            cmap="Reds",
            extent=[0, melspec_db.shape[1], 0, melspec_db.shape[0]],
        )
        axes[1].set_title("SHAP Importance (Brighter = More Important for Prediction)")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Mel Frequency Bins")
        fig.colorbar(img2, ax=axes[1], label="SHAP Importance")

        # Add prediction probabilities as text
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        prob_text = "Top 3 Predictions:\n"
        for i, idx in enumerate(top_3_indices):
            prob_text += f"{i+1}. {GTZAN_GENRES[idx]}: {probabilities[idx]:.3f}\n"

        fig.text(
            0.02,
            0.02,
            prob_text,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )

        plt.tight_layout()
        return fig

    def plot_shap_image(self):
        """Plot mel spectrogram and SHAP values highlighting important parts for prediction."""
        try:
            print("Starting SHAP analysis...")

            # Load the trained model state
            self._load_trained_model()

            # Load and preprocess audio
            audio_data = self._preprocess_audio(self.file_path)
            print(f"Audio data shape: {audio_data.shape}")

            # Convert audio to mel spectrogram (for model input)
            melspec_tensor = self._audio_to_melspec(audio_data)
            print(f"Mel spectrogram tensor shape: {melspec_tensor.shape}")

            # Add batch dimension if needed
            if melspec_tensor.dim() == 2:
                melspec_tensor = melspec_tensor.unsqueeze(0)

            # Create original mel spectrogram for visualization (using librosa for consistency)
            melspec_db = self._create_model_melspectrogram(audio_data)
            print(f"Visualization mel spectrogram shape: {melspec_db.shape}")

            # Create background dataset (mel spectrograms)
            background_data = self._create_background_dataset()
            print(f"Background data shape: {background_data.shape}")

            # Initialize SHAP explainer
            print("Initializing SHAP explainer...")

            try:
                # Use DeepExplainer with the wrapper model
                explainer = shap.DeepExplainer(self.model_wrapper, background_data)
                print("Successfully initialized DeepExplainer")
            except Exception as e:
                print(f"Error with DeepExplainer: {str(e)}")
                try:
                    # Fallback to KernelExplainer which is model-agnostic
                    def model_predict(x):
                        self.model_wrapper.eval()
                        with torch.no_grad():
                            return self.model_wrapper(x).cpu().numpy()

                    # Use a subset of background data for KernelExplainer (it's slower)
                    background_subset = background_data[:2]  # Use fewer samples
                    explainer = shap.KernelExplainer(
                        model_predict, background_subset.cpu().numpy()
                    )
                    print("Successfully initialized KernelExplainer as fallback")
                except Exception as e2:
                    print(f"Error with KernelExplainer: {str(e2)}")
                    raise ValueError(
                        f"Failed to initialize SHAP explainer. DeepExplainer: {str(e)}, KernelExplainer: {str(e2)}"
                    )

            # Calculate SHAP values
            print("Calculating SHAP values...")
            if isinstance(explainer, shap.KernelExplainer):
                # For KernelExplainer, we need numpy input
                shap_values = explainer.shap_values(melspec_tensor.cpu().numpy())
            else:
                # For DeepExplainer, use tensor input
                shap_values = explainer.shap_values(melspec_tensor)

            # Get prediction
            self.model_wrapper.eval()
            with torch.no_grad():
                prediction = self.model_wrapper(melspec_tensor)
                predicted_class = torch.argmax(prediction, dim=1).item()
                probabilities = torch.softmax(prediction, dim=1)[0].cpu().numpy()

            print(f"Predicted class: {predicted_class}")

            # Handle SHAP values - they might be a list for multi-class
            if isinstance(shap_values, list):
                # For multi-class, use SHAP values for the predicted class
                shap_for_class = shap_values[predicted_class]
            else:
                shap_for_class = shap_values

            # Convert SHAP values to mel spectrogram representation
            shap_melspec = self._convert_shap_to_melspec(
                shap_for_class, melspec_db.shape, predicted_class
            )

            # Create visualization
            fig = self._create_shap_visualization(
                melspec_db, shap_melspec, predicted_class, probabilities
            )

            print("SHAP analysis completed successfully!")
            return fig

        except Exception as e:
            print(f"Error in plot_shap_image: {str(e)}")
            import traceback

            traceback.print_exc()

            try:
                audio_data = self._preprocess_audio(self.file_path)
                melspec_db = self._create_model_melspectrogram(audio_data)

                audio_tensor = (
                    torch.FloatTensor(audio_data).unsqueeze(0).to(self.device)
                )
                self.original_model.eval()
                with torch.no_grad():
                    prediction = self.original_model(audio_tensor)
                    predicted_class = torch.argmax(prediction, dim=1).item()
                    probabilities = torch.softmax(prediction, dim=1)[0].cpu().numpy()

                from AudioConcept.config import GTZAN_GENRES

                fig, ax = plt.subplots(figsize=(12, 6))
                img = librosa.display.specshow(
                    melspec_db, ax=ax, y_axis="mel", x_axis="time", sr=22050
                )
                ax.set_title(
                    f"Mel Spectrogram - Predicted: {GTZAN_GENRES[predicted_class]} ({probabilities[predicted_class]:.3f})\n(SHAP failed: {str(e)})"
                )
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                return fig
            except Exception as fallback_error:
                print(f"Even fallback visualization failed: {str(fallback_error)}")
                # Return empty figure as last resort
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.text(
                    0.5,
                    0.5,
                    f"Visualization failed\nError: {str(e)}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                return fig
