import os
import random
import torch
import numpy as np
import soundfile as sf
from torch.utils import data
from config import DATA_PATH

# from torchaudio_augmentations import (
#     RandomResizedCrop,
#     RandomApply,
#     PolarityInversion,
#     Noise,
#     Gain,
#     HighLowPass,
#     Delay,
#     PitchShift,
#     Reverb,
#     Compose,
# )

GTZAN_GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


class GTZANDataset(data.Dataset):
    def __init__(self, data_path, split, num_samples, num_chunks, is_augmentation):
        self.data_path = data_path if data_path else ""
        self.split = split
        self.num_samples = num_samples
        self.num_chunks = num_chunks
        self.is_augmentation = is_augmentation
        self.genres = GTZAN_GENRES
        self._get_song_list()
        if is_augmentation:
            self._get_augmentations()

    def _get_song_list(self):
        list_filename = os.path.join(self.data_path, "%s_filtered.txt" % self.split)
        with open(list_filename) as f:
            lines = f.readlines()
        self.song_list = [line.strip() for line in lines]

    def _get_augmentations(self):
        # transforms = [
        #     RandomResizedCrop(n_samples=self.num_samples),
        #     RandomApply([PolarityInversion()], p=0.8),
        #     RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
        #     RandomApply([Gain()], p=0.2),
        #     RandomApply([HighLowPass(sample_rate=22050)], p=0.8),
        #     RandomApply([Delay(sample_rate=22050)], p=0.5),
        #     RandomApply(
        #         [PitchShift(n_samples=self.num_samples, sample_rate=22050)], p=0.4
        #     ),
        #     RandomApply([Reverb(sample_rate=22050)], p=0.3),
        # ]
        # self.augmentation = Compose(transforms=transforms)
        pass

    def _adjust_audio_length(self, wav):
        """
        Random chunks of audio are cropped from the entire sequence during the
        training. But in validation / test phase, an entire sequence is split
        into multiple chunks and the chunks are stacked. The stacked chunks are
        later input to a trained model and the output predictions are aggregated
        to make song-level predictions.
        """
        if self.split == "train":
            random_index = random.randint(0, len(wav) - self.num_samples - 1)
            wav = wav[random_index : random_index + self.num_samples]
        else:
            hop = (len(wav) - self.num_samples) // self.num_chunks
            wav = np.array(
                [
                    wav[i * hop : i * hop + self.num_samples]
                    for i in range(self.num_chunks)
                ]
            )
        return wav

    def __getitem__(self, index):
        line = self.song_list[index]

        # get genre
        genre_name = line.split("/")[0]
        genre_index = self.genres.index(genre_name)

        # get audio
        audio_filename = os.path.join(self.data_path, "genres", line)
        wav, fs = sf.read(audio_filename)

        # adjust audio length
        wav = self._adjust_audio_length(wav).astype("float32")

        # data augmentation
        if self.is_augmentation:
            wav = (
                self.augmentation(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
            )

        return wav, genre_index

    def __len__(self):
        return len(self.song_list)


def get_dataloader(
    data_path=DATA_PATH,
    split="train",
    num_samples=22050 * 29,
    num_chunks=1,
    batch_size=16,
    num_workers=0,
    is_augmentation=False,
):
    is_shuffle = True if (split == "train") else False
    batch_size = batch_size if (split == "train") else (batch_size // num_chunks)
    data_loader = data.DataLoader(
        dataset=GTZANDataset(
            data_path, split, num_samples, num_chunks, is_augmentation
        ),
        batch_size=batch_size,
        shuffle=is_shuffle,
        drop_last=False,
        num_workers=num_workers,
    )
    return data_loader


train_loader = get_dataloader(split="train", is_augmentation=False)
valid_loader = get_dataloader(split="valid")
test_loader = get_dataloader(split="test")


if __name__ == "__main__":
    iter_train_loader = iter(train_loader)
    train_wav, train_genre = next(iter_train_loader)
    iter_test_loader = iter(test_loader)
    test_wav, test_genre = next(iter_test_loader)
    print("training data shape: %s" % str(train_wav.shape))
    print("training targets: ", train_genre)
    print("validation/test data shape: %s" % str(test_wav.shape))
    print(train_genre)
