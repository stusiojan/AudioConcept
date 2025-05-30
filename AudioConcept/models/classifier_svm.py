import pickle
import wandb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from AudioConcept.config import GTZAN_GENRES, SVM_PARAM_GRID
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class SVMClassifier:
    """SVM Classifier with grid search and wandb logging.

    From paper:
    G. ARCHITECTURE OF SVM MODEL
    In our study, the SVM architecture is described in the obtained
    Mel-Frequency Cepstral Coefficients (MFCC) features of au-
    dio samples for music genre classification. Initially, the audio
    files are sampled at a rate of 22,050 Hz, and each track is
    divided into numerous parts lasting 30 seconds each. MFCC
    features are then computed for each segment via the librosa
    package. The generated MFCC vectors are flattened into one-
    dimensional arrays and divided into training and testing sets.
    Changsheng et al. [32] propose effective algorithms to auto-
    matically classify and summarize music content, and SVM is
    used to classify music. Selecting the hyperplane in the feature
    space that best divides several classes is the foundation of
    SVMs, as opposed to neural networks. A linear SVM model
    is trained on the training set using the sci-kit-learn library’s
    SVC class, with the regularization parameter (C) set to one.
    Subsequently, the trained model is used to predict genre labels
    for the test set. The accuracy-score function from sci-kit-learn
    is used to evaluate the SVM model’s performance in classi-
    fying music genres. This architecture shows the process of
    feature extraction, model training, prediction, and evaluation
    in SVM-based music genre classification, with a focus on the
    audio sample rate and track duration parameters.

    According to study the best results should be achieved with:
    kernel = linear
    C = 1
    """

    def __init__(self, experiment_name="svm_classifier", use_wandb=True):
        """Initialize the classifier.

        Args:
            experiment_name: Name for the wandb experiment
            use_wandb: Whether to use wandb logging
        """
        self.model = None
        self.scaler = StandardScaler()
        self.experiment_name = experiment_name
        self.best_params_ = None
        self.use_wandb = use_wandb
        self.genres = GTZAN_GENRES

        # Parameters for grid search
        self.param_grid = SVM_PARAM_GRID

    def train(self, model_path, X, y, random_state, cv=5):
        """Train the model using grid search and cross-validation.

        Args:
            models_dir: Directory to save the model
            X: Feature matrix
            y: Target labels
            random_state: seed
            cv: Number of cross-validation folds
        """
        try:
            if self.use_wandb:
                try:
                    wandb.init(project="audio-concept", name=self.experiment_name)
                except Exception as e:
                    logger.warning(f"Failed to initialize wandb: {e}")
                    logger.warning("Continuing without wandb logging...")
                    self.use_wandb = False

            X_scaled = self.scaler.fit_transform(X)

            svm = SVC(random_state=random_state)

            grid_search = GridSearchCV(
                svm,
                self.param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1,  # Add progress reporting
            )
            grid_search.fit(X_scaled, y)

            # Store best model and parameters
            # might and propably will differ from best SVM from the paper
            self.model = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_

            logger.info(f"Best parameters: {self.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_}")

            if self.use_wandb:
                try:
                    table = wandb.Table(
                        columns=["Parameter", "Value"],
                        data=[
                            [f"{param}", f"{grid_search.best_params_[param]}"]
                            for param in grid_search.best_params_
                        ],
                    )
                    wandb.config.update(grid_search.best_params_)
                    wandb.log({"grid_search_results": table})

                except Exception as e:
                    logger.warning(f"Failed to log to wandb: {e}")

            self.save_model(model_path)

            return grid_search.best_score_

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def predict(self, X):
        """Make predictions on new data.

        Args:
            X: Feature matrix
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X, y):
        """Evaluate model performance.

        Args:
            X: Feature matrix
            y: True labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        y_pred = self.predict(X)

        accuracy = accuracy_score(y, y_pred)
        report = classification_report(
            y, y_pred, target_names=self.genres, output_dict=True
        )
        conf_matrix = confusion_matrix(y, y_pred)

        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Report: {report}")
        logger.info(f"Confusion matrix: {conf_matrix}")

        if self.use_wandb:
            try:
                table = wandb.Table(
                    columns=["Genre", "Precision", "Recall", "F1-Score"],
                    data=[
                        [
                            genre,
                            report[genre]["precision"],
                            report[genre]["recall"],
                            report[genre]["f1-score"],
                        ]
                        for genre in self.genres
                    ],
                )
                # wandb.log({"test_accuracy": accuracy, "classification_report": report})
                wandb.log({"classification_report_table": table})
                wandb.log({"confusion_matrix": wandb.Image(conf_matrix)})
                logger.info("Accuracy logged")
            except Exception as e:
                logger.warning(f"Failed to log accuracy metrics to wandb: {str(e)}")

        return accuracy, report, conf_matrix

    def save_model(self, model_path):
        """Save the trained model to disk."""
        saved_model_path = model_path
        with open(saved_model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "best_params": self.best_params_,
                },
                f,
            )
        logger.info(f"Model saved to {saved_model_path}")

    def load_model(self, model_path):
        """Load a trained model from disk.

        Args:
            model_name: Name of the model file (without .pkl extension)
            models_dir: Directory containing the model file
        """
        load_model_path = model_path
        with open(load_model_path, "rb") as f:
            saved_data = pickle.load(f)
            self.model = saved_data["model"]
            self.scaler = saved_data["scaler"]
            self.best_params_ = saved_data["best_params"]
        logger.info(f"Model loaded from {load_model_path}")

    # Plotting
    def plot_confusion_matrix(self, confusion_matrix, save_path):
        """Plot confusion matrix.

        Args:
            confusion_matrix: Confusion matrix from model evaluation
            genres: List of genre names
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(15, 12))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.genres,
            yticklabels=self.genres,
        )
        plt.title("Confusion Matrix - All Genres")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        try:
            wandb.log(
                {
                    "confusion_matrix": wandb.Image(plt.gcf()),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log confusion matrix to wandb: {e}")

        plt.close()

    def plot_grid_search_results(cv_results, param_name, save_path):
        """Plot grid search results for a specific parameter.

        Args:
            cv_results: CV results from GridSearchCV
            param_name: Name of the parameter to plot
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))

        mean_scores = np.array(cv_results["mean_test_score"])
        std_scores = np.array(cv_results["std_test_score"])

        if param_name == "kernel":
            x_values = range(len(cv_results[f"param_{param_name}"]))
            plt.xticks(x_values, cv_results[f"param_{param_name}"])
        else:
            x_values = np.array(cv_results[f"param_{param_name}"])

        plt.plot(x_values, mean_scores, "o-")
        plt.fill_between(
            range(len(cv_results[f"param_{param_name}"])),
            mean_scores - std_scores,
            mean_scores + std_scores,
            alpha=0.2,
        )

        plt.xlabel(param_name)
        plt.ylabel("CV Score")
        plt.title(f"Grid Search Results - {param_name}")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        try:
            wandb.log(
                {
                    f"grid_search_{param_name}": wandb.Image(plt.gcf()),
                    f"grid_search_{param_name}_data": {
                        "param_values": cv_results[f"param_{param_name}"],
                        "mean_scores": mean_scores.tolist(),
                        "std_scores": std_scores.tolist(),
                    },
                }
            )
        except Exception as e:
            logger.warning(f"Failed to log grid search results to wandb: {e}")

        plt.close()
