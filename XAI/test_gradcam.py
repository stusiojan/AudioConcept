# import torch
# from AudioConcept.models.model_cnn import CNN
# from XAI.cnn_gradcam import GradCAM, plot_gradcam

# # 1. Załaduj model
# model = CNN()
# model = torch.load("../models/best_CNN_model.pkl", map_location=torch.device('cpu'), weights_only=False)


# # 2. Przygotuj przykładowy input
# wav = torch.randn(1, 22050)  # tu wrzuć swój dźwięk
# mel = model.melspec(wav)
# mel_db = model.amplitude_to_db(mel).unsqueeze(1)  # (1, 1, H, W)

# # 3. Wybierz ostatnią warstwę konwolucyjną (np. layer5.conv)
# gradcam = GradCAM(model, model.layer5.conv)

# # 4. Generuj i pokaż
# heatmap = gradcam.generate(wav)
# plot_gradcam(heatmap, mel_db.squeeze().numpy())


import torch

model = torch.load("models/best_CNN_model.pkl", map_location="cpu", weights_only=False)
torch.save(model.state_dict(), "best_CNN_model.ckpt")


