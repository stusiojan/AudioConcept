
# 🔬 Experiments

## VGGish

### Model architecture

VGG 16 convolution architecture was introduced in 2015([original paper](https://arxiv.org/pdf/1409.1556)) and according to authors was "network of increasing depth using an architecture with very small (3× 3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16–19 weight layers."


![](../reports/vgg_arch_researchgate.webp)
*VGG 16 architecture visualization. Source: researchgate.net*

In 2017 [paper](https://arxiv.org/pdf/1609.09430) there was a attempt to use VGG 19 in music classification tasks using spectrograms as an audio data representation it was called VGGish. The model were trained on YouTube-100M dataset. They made a minor changes to the architecture: "the final layer (3087 units with a sigmoid) as well as the use of batch normalization instead of LRN [...] We tried another variant that reduced the initial strides (as we did with AlexNet), but found that not modifying the
strides resulted in faster training and better performance."

Here we implemented the VGGish architecture with changes:
- 16 instead of 19 layers due to much smaller dataset (less than 1000 samples compared to 100 000 000)
- the last layer is a fully connected layer without sigmoid (the final layer - 3087 units with a sigmoid the results were [worse](#final-layer-experiments))

### GTZAN Dataset

We have experiment with traning on 29.1 and 3.96 seconds long audio. Network trains faster on shorter data and gives similar results, given that evaluation is on same length data. If we train on shorter and evaluate on longer data, ther results go down about 40% (accuracy drop from 0.48 to 0.29). The other way around the drop is similar.

We used following augementations:
```python
RandomResizedCrop(n_samples=self.num_samples),
RandomApply([PolarityInversion()], p=0.5),
RandomApply([Noise(min_snr_db=15, max_snr_db=35)], p=0.6),
RandomApply([Gain(min_gain_db=-4, max_gain_db=4)], p=0.4),
RandomApply([HighLowPass(sample_rate=self.sample_rate)], p=0.5),
RandomApply([Delay(sample_rate=self.sample_rate)], p=0.3),
RandomApply([TimeStretch(n_samples=self.num_samples)], p=0.2)
```

### Learning rate experiments

This was the base configuration for this experimensts:
```
NUM_EPOCHS = 60
MODEL_PATIENCE = 15
WEIGHT_DECAY = 8e-4
LABEL_SMOOTHING = 0.15
NOISE_LEVEL = 0.002
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.75,
    patience=8,
    min_lr=effective_lr / 50,
    verbose=True,
)
AUDIO_LENGTH=3.96
```
the data augemntation and model architecture were the same.

LEARNING RATE = 8e-6
![](../reports/figures/VGGish_training_plots_20250531_2206.png)
LEARNING RATE = 1e-5
![](../reports/figures/VGGish_training_plots_20250531_2303.png)
LEARNING RATE = 3e-5
![](../reports/figures/VGGish_training_plots_20250531_2358.png)
LEARNING RATE = **4e-5** (the highest accuracy)
![](../reports/figures/VGGish_training_plots_20250531_0202.png)
LEARNING RATE = 5e-5
![](../reports/figures/VGGish_training_plots_20250531_2021.png)
LEARNING RATE = 6e-5
![](../reports/figures/VGGish_training_plots_20250531_2118.png)


### weight decay experiments

WEIGHT DECAY 8e-6
![](../reports/figures/VGGish_training_plots_20250601_0159.png)
WEIGHT DECAY **5e-5** (the highest accuracy)
![](../reports/figures/VGGish_training_plots_20250601_0246.png)
WEIGHT DECAY 1e-4
![](../reports/figures/VGGish_training_plots_20250601_0342.png)
WEIGHT DECAY  8e-4
![](../reports/figures/VGGish_training_plots_20250601_0433.png)
WEIGHT DECAY  1e-3
![](../reports/figures/VGGish_training_plots_20250601_0513.png)
WEIGHT DECAY  1e-2
![](../reports/figures/VGGish_training_plots_20250601_0603.png)


Experiments for the shorter batch:

WEIGHT DECAY = 1e-4
![](../reports/figures/VGGish_training_plots_20250528_1833.png)
WEIGHT DECAY  = 1e-3
![](../reports/figures/VGGish_training_plots_20250528_1852.png)
WEIGHT DECAY  = **5e-4** (the highest accuracy)
![](../reports/figures/VGGish_training_plots_20250528_2006.png)
WEIGHT DECAY  = 5e-3 
![](../reports/figures/VGGish_training_plots_20250528_2028.png)

*NOTE: the best results were ofter achieved in the 9th epoc. This was propably cause due to high ReduceLROnPlateau scheduler patience*

### Final layer experiments

Results of training on the same data with the same parameters but different architectures (ours VGGish implementation and with added 3087 units final layer with sigmoid)

1. without sigmoid
![](../reports/figures/VGGish_training_plots_20250601_1918.png)
2. with sigmoid
![](../reports/figures/VGGish_training_plots_20250601_1855.png)

## CNN

### Model architecture

The CNN model architecture is based on a convolutional neural network approach specifically designed for audio classification tasks. This implementation draws from the work by mamodrzejewski's GTZAN genre classification [example](https://github.com/mamodrzejewski/wimu-gtzan-genre-example) and the tutorial "Music Classification: Beyond Supervised Learning, Towards Real-world Applications" by Won, Spijkervet and Choi.

The architecture consists of:

1. **Audio Preprocessing Layer**
    - Mel Spectrogram transformation with 128 mel bands
    - Amplitude to dB conversion
    - Input batch normalization

2. **Convolutional Feature Extraction**
    - Five convolutional blocks with increasing channel dimensions (16 → 16 → 32 → 32 → 64)
    - Each block contains:
      - 2D convolution with 3×3 kernels
      - Batch normalization
      - ReLU activation
      - Max pooling with varying pool sizes optimized for audio spectrograms
      - Dropout (0.1) for regularization

3. **Classification Head**
    - Flattened convolutional features
    - Dense layer maintaining feature dimensionality (64 → 64)
    - Batch normalization
    - ReLU activation
    - Dropout (0.5) for stronger regularization
    - Final dense layer mapping to class probabilities (64 → 10)

This architecture effectively processes the time-frequency representation of audio signals, with the convolutional layers progressively extracting more complex audio patterns while reducing the spatial dimensions. The varying pooling sizes are specifically designed to handle the non-uniform importance of time and frequency dimensions in audio spectrograms.

### GTZAN Dataset

The network was trained on 29.1 seconds long audio from GTZAN dataset with following augmentations:
```python
RandomResizedCrop(n_samples=self.num_samples),
RandomApply([PolarityInversion()], p=0.5),
RandomApply([Noise(min_snr_db=15, max_snr_db=35)], p=0.6),
RandomApply([Gain(min_gain_db=-4, max_gain_db=4)], p=0.4),
RandomApply([HighLowPass(sample_rate=self.sample_rate)], p=0.5),
RandomApply([Delay(sample_rate=self.sample_rate)], p=0.3),
RandomApply([TimeStretch(n_samples=self.num_samples)], p=0.2)
```

### Learning rate experiments

This was the base configuration for this experimensts:
```
NUM_EPOCHS = 60
MODEL_PATIENCE = 15
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.15
NOISE_LEVEL = 0.002
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.75,
    patience=8,
    min_lr=effective_lr / 50,
    verbose=True,
)
AUDIO_LENGTH=29.1
```
the data augemntation and model architecture were the same.

LEARNING RATE = 8e-2
![](../reports/figures/CNN_training_plots_20250601_0659.png)

LEARNING RATE = 1e-3
![](../reports/figures/CNN_training_plots_20250601_0736.png)

LEARNING RATE = **5e-3** (the highest accuracy)
![](../reports/figures/CNN_training_plots_20250601_0857.png)

LEARNING RATE = 7e-3
![](../reports/figures/CNN_training_plots_20250601_1019.png)

LEARNING RATE = 9e-3
![](../reports/figures/CNN_training_plots_20250601_1136.png)

LEARNING RATE = 5e-4
![](../reports/figures/CNN_training_plots_20250601_1254.png)

LEARNING RATE = 9e-4
![](../reports/figures/CNN_training_plots_20250601_1415.png)

With the biggest learning rate from tested ones the model doesn't learn.
With the lowest learning rates the model is overfitting. This learning rates could be better if we use more agressive regaluration and enlargen the dataset.

### LLM experiment

The task of training CNN on GTZAN dataset is to classify audio genres is a common theme in audio processing, so we tried to fully generate
training pipeline with LLM.

We did not spend much time on experimenting with this neural network. The best training session resulted with 32.9 validation accuracy. Training parameters:
```
epochs = 20
batch_size = 16
learning_rate = 0.0005

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(
    self.model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8,
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=args.patience,
    verbose=True,
    min_lr=args.min_lr / 10,
)
```

![](../reports/llm_training_curves.png)

## SVM

We experimented with this parameters with full feature extraction:
```
SVM_TEST_SIZE = 0.2  # Proportion of dataset to include in the test split
SVM_RANDOM_STATE = 42
SVM_PARAM_GRID = {
    "C": [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto", 0.1, 5e-3, 0.01, 0.1, 1],
}
SVM_FEATURES_FILTER = [
    "length",
    "chroma_stft_mean",
    "chroma_stft_var",
    "rms_mean",
    "rms_var",
    "spectral_centroid_var",
    "spectral_bandwidth_var",
    "rolloff_mean",
    "rolloff_var",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_var",
    "harmony_mean",
    "harmony_var",
    "perceptr_mean",
    "perceptr_var",
    "tempo",
    "mfcc1_mean",
    "mfcc1_var",
    "mfcc2_var",
    "mfcc3_mean",
    "mfcc3_var",
    "mfcc4_mean",
    "mfcc4_var",
    "mfcc5_mean",
    "mfcc5_var",
    "mfcc6_mean",
    "mfcc6_var",
    "mfcc7_mean",
    "mfcc7_var",
    "mfcc8_mean",
    "mfcc8_var",
    "mfcc9_mean",
    "mfcc9_var",
    "mfcc10_mean",
    "mfcc10_var",
    "mfcc11_mean",
    "mfcc11_var",
    "mfcc12_mean",
    "mfcc12_var",
    "mfcc13_mean",
    "mfcc13_var",
    "mfcc14_mean",
    "mfcc14_var",
    "mfcc15_mean",
    "mfcc15_var",
    "mfcc16_mean",
    "mfcc16_var",
    "mfcc17_mean",
    "mfcc17_var",
    "mfcc18_mean",
    "mfcc18_var",
    "mfcc19_mean",
    "mfcc19_var",
    "mfcc20_mean",
    "mfcc20_var",
]
```

We achieved the best results for:
- C = 5
- gamma = 0.01
- kernel = rbf
with random state 42,
resulting in accuracy: 0.76

According to [study]() the best results should be achieved with:
- kernel = linear
- C = 1
resulting in accuracy: 0.79

### Feature selection

We use SHAP library to see which features has the most impact on classification and trained only on them.

Features that most often has the biggest impact:
```
"chroma_stft_mean",
"perceptr_var",
"mfcc4_mean",
"mfcc5_var",
"mfcc6_mean",
"mfcc7_var",
"mfcc9_mean",
"mfcc9_var",
"mfcc12_var",
"mfcc13_var",
"mfcc14_mean",
"mfcc14_var",
"mfcc15_mean",
"mfcc17_mean",
```

This resulted with decreased accuracy of 0.64 with best model for parameters:
- C = 2
- gamma = auto
- kernel = rbf
with random state 42

Exemplary results (slightly different features than above):
![](../reports/features.png)

## 💡 Conclusions

For GTZAN data, which is quite small dataset, complicated architectures are no better than simpler ones. We expected significant difference between VGG and CNN. The scores similarity can be attributed to both models suffering from the same fundamental limitation - insufficient training data to leverage their representational capacity, leading to overfitting and poor generalization regardless of architectural complexity.

Neural network architectures were outperformed by SVM. According to literature it should be other way around. This could be caused by the effectiveness of audio features used by SVM that capture years of domain expertise, and the neural networks tendency to overfit on small datasets despite regularization attempts.

## 🗓️ Future work

Couple training process with XAI methods, so we can be more precisely address classification weak points.

We can extend dataset and add more augmentations.

Implement residual architecture for better gradient flow and see if the result will improve.