import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage
from AudioConcept.models.model_cnn import CNN

class GradCAM:
    def __init__(self, model: CNN, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        # Wyzeruj gradienty
        self.model.zero_grad()

        # Wsteczna propagacja dla danej klasy
        loss = output[0, class_idx]
        loss.backward()

        # Średnie gradienty po przestrzeni (Global Average Pooling)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Wzmocnij aktywacje gradientami
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # Grad-CAM: sumuj i ReLU
        heatmap = torch.sum(activations, dim=0)
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap) + 1e-8
        return heatmap.cpu().numpy()


def plot_gradcam(heatmap, original_mel, save_path=None):
    plt.figure(figsize=(10, 4))
    plt.imshow(original_mel, aspect='auto', origin='lower')
    plt.imshow(heatmap, cmap='jet', alpha=0.4, extent=(0, original_mel.shape[1], 0, original_mel.shape[0]))
    plt.colorbar(label='Importance')
    plt.title("Grad-CAM Visualization")
    if save_path:
        plt.savefig(save_path)
    plt.show()
