import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from model import CNN
from gtzan_loader import test_loader, GTZAN_GENRES
from datetime import datetime

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = CNN().to(device)

    # Load the best model
    # S = torch.load("models/best_model_backup.ckpt")
    S = torch.load("models/best_model.ckpt")
    cnn.load_state_dict(S)
    print("loaded!")

    # Run evaluation
    cnn.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for wav, genre_index in test_loader:
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            # reshape and aggregate chunk-level predictions
            b, c, t = wav.size()
            logits = cnn(wav.view(-1, t))
            logits = logits.view(b, c, -1).mean(dim=1)
            _, pred = torch.max(logits.data, 1)

            # append labels and predictions
            y_true.extend(genre_index.tolist())
            y_pred.extend(pred.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    confusion_matrix = sns.heatmap(
        cm,
        annot=True,
        xticklabels=GTZAN_GENRES,
        yticklabels=GTZAN_GENRES,
        cmap="YlGnBu",
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plt.savefig(f"results/confusion_matrix_{timestamp}.png", dpi=300, bbox_inches="tight")
    print("Accuracy: %.4f" % accuracy)

    # plt.show()
