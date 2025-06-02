from XAI.xai_waterfall import XaiWaterfall


def generate_waterfall_plot():
    """Generuje wykres waterfall na podstawie modelu SHAP."""
    xai = XaiWaterfall()
    xai.load_data()
    xai.load_model()
    xai.scale_data()
    xai.compute_shap_values()
    fig = xai.plot_waterfall()
    return fig
