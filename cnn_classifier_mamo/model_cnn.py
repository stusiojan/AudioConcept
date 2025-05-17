from torch import nn
import torchaudio


class Conv_2d(nn.Module):
    def __init__(
        self, input_channels, output_channels, shape=3, pooling=2, dropout=0.1
    ):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, shape, padding=shape // 2
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pooling)
        self.dropout = nn.Dropout(dropout)

    def forward(self, wav):
        out = self.conv(wav)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        return out


class CNN(nn.Module):
    def __init__(
        self,
        num_channels=16,
        sample_rate=22050,
        n_fft=1024,
        f_min=0.0,
        f_max=11025.0,
        num_mels=128,
        num_classes=10,
    ):
        super(CNN, self).__init__()

        # mel spectrogram
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=num_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.input_bn = nn.BatchNorm2d(1)

        # convolutional layers
        self.layer1 = Conv_2d(1, num_channels, pooling=(2, 3))
        self.layer2 = Conv_2d(num_channels, num_channels, pooling=(3, 4))
        self.layer3 = Conv_2d(num_channels, num_channels * 2, pooling=(2, 5))
        self.layer4 = Conv_2d(num_channels * 2, num_channels * 2, pooling=(3, 3))
        self.layer5 = Conv_2d(num_channels * 2, num_channels * 4, pooling=(3, 4))

        # dense layers
        self.dense1 = nn.Linear(num_channels * 4, num_channels * 4)
        self.dense_bn = nn.BatchNorm1d(num_channels * 4)
        self.dense2 = nn.Linear(num_channels * 4, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, wav):
        # input preprocessing
        out = self.melspec(wav)
        print(f"Mel spectrogram shape: {out.shape}")
        out = self.amplitude_to_db(out)
        print(f"Amplitude to dB shape: {out.shape}")

        # input batch normalization
        out = out.unsqueeze(1)
        out = self.input_bn(out)
        print(f"Input batch normalization shape: {out.shape}")

        # convolutional layers
        out = self.layer1(out)
        print(f"Convolutional layer 1 shape: {out.shape}")
        out = self.layer2(out)
        print(f"Convolutional layer 2 shape: {out.shape}")
        out = self.layer3(out)
        print(f"Convolutional layer 3 shape: {out.shape}")
        out = self.layer4(out)
        print(f"Convolutional layer 4 shape: {out.shape}")
        out = self.layer5(out)
        print(f"Convolutional layer 5 shape: {out.shape}")

        # reshape (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
        out = out.reshape(len(out), -1)

        # dense layers
        out = self.dense1(out)
        out = self.dense_bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.dense2(out)
        print(f"Dense layer output shape: {out.shape}")

        # example output shapes
        # Mel spectrogram shape: torch.Size([16, 128, 1249])
        # Amplitude to dB shape: torch.Size([16, 128, 1249])
        # Input batch normalization shape: torch.Size([16, 1, 128, 1249])
        # Convolutional layer 1 shape: torch.Size([16, 16, 64, 416])
        # Convolutional layer 2 shape: torch.Size([16, 16, 21, 104])
        # Convolutional layer 3 shape: torch.Size([16, 32, 10, 20])
        # Convolutional layer 4 shape: torch.Size([16, 32, 3, 6])
        # Convolutional layer 5 shape: torch.Size([16, 64, 1, 1])
        # Dense layer output shape: torch.Size([16, 10])
        return out
