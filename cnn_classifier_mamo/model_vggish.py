from torch import nn
from config import VGG16_ARCHITECTURE
import torchaudio


class VGGish(nn.Module):
    def __init__(
        self,
        num_channels=1,  # 16,
        sample_rate=22050,
        n_fft=1024,
        f_min=0.0,
        f_max=11025.0,
        num_mels=128,
        num_classes=10,
    ):
        super(VGGish, self).__init__()

        self.in_channels = num_channels

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
        self.conv_layers = self.create_conv_layers(VGG16_ARCHITECTURE)

        # fully connected layers
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 7 = input size / 2^num_maxpool = 224 / 2^5
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                print(f"Input channels: {in_channels}")
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),  # might be the same as kernel_size
                    ),
                    nn.BatchNorm2d(x),  # Not included in the original paper
                    nn.ReLU(),
                    # maxpool, dropout - not included in the original paper
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

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

        # Reshape from [batch, 1, 128, 1249] to [batch, 1, 224, 224]
        out = nn.functional.interpolate(
            out, size=(224, 224), mode="bilinear", align_corners=False
        )
        print(f"After reshaping to 224x224: {out.shape}")

        # # expand to 16 channels (repeat the same data across channels)
        # out = out.repeat(1, self.in_channels, 1, 1)
        # print(f"After channel expansion shape: {out.shape}")
        # out here should be in this format: torch.randn(n, 16, 224, 224).to(device)

        # convolutional layers
        out = self.conv_layers(out)
        print(f"Convolutional layers shape: {out.shape}")

        # reshape (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
        # out = out.reshape(out.shape[0], -1)
        out = out.reshape(len(out), -1)
        print(f"Reshaped output shape: {out.shape}")

        # fully connected layers
        out = self.fcs(out)
        print(f"Fully connected layers output shape: {out.shape}")
        # example output shapes
        # Mel spectrogram shape: torch.Size([16, 128, 1249])
        # Amplitude to dB shape: torch.Size([16, 128, 1249])
        # Input batch normalization shape: torch.Size([16, 1, 128, 1249])
        # Convolutional layers shape: torch.Size([16, 512, 4, 39])
        # Reshaped output shape: torch.Size([16, 79872])

        return out
