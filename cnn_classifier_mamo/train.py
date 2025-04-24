import numpy as np
import torch
from torch import nn
from model import CNN
from gtzan_loader import train_loader, valid_loader
from sklearn.metrics import accuracy_score
from datetime import datetime
from config import LEARNING_RATE, NUM_EPOCHS
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # Tensorboard setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = f"GTZAN_CNN_{timestamp}_LR_{LEARNING_RATE}"
    writer = SummaryWriter(f"runs/{experiment_name}")

    # CNN setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = CNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    valid_losses = []
    num_epochs = NUM_EPOCHS

    for epoch in range(num_epochs):
        losses = []

        # train
        cnn.train()
        for batch_idx, (wav, genre_index) in enumerate(train_loader):
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            # forward
            out = cnn(wav)
            loss = loss_function(out, genre_index)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Training Loss", loss.item(), global_step)

            losses.append(loss.item())

        print(
            "Epoch: [%d/%d], Train loss: %.4f"
            % (epoch + 1, num_epochs, np.mean(losses))
        )

        # Validation
        cnn.eval()
        y_true = []
        y_pred = []
        losses = []
        for wav, genre_index in valid_loader:
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            # reshape and aggregate chunk-level predictions
            b, c, t = wav.size()
            logits = cnn(wav.view(-1, t))
            logits = logits.view(b, c, -1).mean(dim=1)
            loss = loss_function(logits, genre_index)
            losses.append(loss.item())
            _, pred = torch.max(logits.data, 1)

            # append labels and predictions
            y_true.extend(genre_index.tolist())
            y_pred.extend(pred.tolist())
        accuracy = accuracy_score(y_true, y_pred)
        valid_loss = np.mean(losses)
        print(
            "Epoch: [%d/%d], Valid loss: %.4f, Valid accuracy: %.4f"
            % (epoch + 1, num_epochs, valid_loss, accuracy)
        )

        # Save model
        valid_losses.append(valid_loss.item())
        if np.argmin(valid_losses) == epoch:
            print("Saving the best model at %d epochs!" % epoch)
            torch.save(cnn.state_dict(), "models/best_model.ckpt")

    writer.close()
