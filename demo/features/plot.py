from XAI.xaiWaterfall import xaiWaterfall
from models.model_type import ModelType


def generate_waterfall_plot(model_choice: str):
    """Generuje wykres waterfall na podstawie modelu SHAP."""

    def _get_model_type(model_choice: str):
        """Zwraca typ modelu na podstawie wyboru użytkownika."""
        if model_choice == ModelType.SVM.value:
            return ModelType.SVM
        elif model_choice == ModelType.RF.value:
            return ModelType.RF
        elif model_choice == ModelType.XGB.value:
            return ModelType.XGB
        else:
            raise ValueError(f"Wrong model name: {model_choice}")

    match _get_model_type(model_choice):
        case ModelType.SVM:
            xai = xaiWaterfall()
            xai.load_data()
            xai.load_model()
            xai.scale_data()
            xai.compute_shap_values()
        case ModelType.CNN:
            pass
        case ModelType.VGGish:
            pass
        case _:
            raise ValueError(f"Unsupported model type: {model_choice}")
    fig = xai.plot_waterfall()
    return fig
