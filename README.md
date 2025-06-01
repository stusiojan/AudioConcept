# AudioConcept

![Version](https://img.shields.io/badge/AudioConcept-1.0.0-orange)
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>


## 📝 Table of contents
- [About](#about)
- [Project Organization](#organization)
- [Setup](#setup)
- [Usage](#usage)
- [Sources](#sources)


<h2 id="about">🧐 About</h2>

Locating Musical Concepts in a Genre Classifier.

<h2 id="organization">🗂️ Project Organization</h2>

```
├── LICENSE            <- Open-source license.
├── Makefile           <- Makefile with convenience commands.
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw
│       ├── gtzan           <- GTZAN dataset without jazz.00054.wav file.
│       └── sample_audio    <- Audio for predictions.
│
├── docs               <- Project documentation.
│
├── models             <- Trained models. Naming convention is "best_{model_name}_model.pkl",
│                         available model names are: SVG, CNN, VGGish.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         AudioConcept and configuration for linting and formatting tools.
│
├── reports            <- Generated analysis.
│   └── figures        <- Generated graphics and figures to be used in reporting.
│
├── requirements.txt   <- The requirements file for reproducing the analysis virtual environment.
│
├── environment.yml    <- The requirements file for reproducing the analysis environment using conda.
│
├── scripts            <- Various automation scripts.
│
└── AudioConcept   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes AudioConcept a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Data prepocessing and testing
    │
    ├── predict.py              <- Code to run model inference with trained models 
    │
    ├── features.py             <- Code to extract features
    │
    ├── train.py                <- Code to train models
    │
    ├── evaluate.py             <- Code to evaluate models
    │
    ├── augmentation.py         <- Code to augment audio data
    │
    ├── llm_experiment          <- LLM generated training pipeline
    │
    └── models                  <- Neural network models and classifiers
```

<h2 id="setup">🚀 Setup</h2>


### Prepare virtual environment
#### Using venv - 18.05 NOT UPDATED
```bash
python 3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Using conda
Install environment (from `environment.yml` on linux, `environment_mac.yml` on macOS)
TODO: Check if it is working on Windows too
```bash
conda env create -f environment_mac.yml
conda activate wimu
```

### WandB

SVM Classifier is using *Weights and biases* for monitoring experiments.

Log in to Weights and biases
```bash
wandb login
# paste your API key from website https://wandb.ai/home
```

### Prepare data

1. Download GTZAN [dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download-directory) and set path to it in `config.py`
2. Rename *genres_original* to *gtzan*. *Images_original* will not be used.
3. Run get data in GTZAN dataset repository
```bash
cd scripts
./get_data.sh
```
4. Make sure that folder names are correct
```txt
gtzan
|-- genres
|   |-- blues
|   ...
|   |-- rock
| test_filtered.txt
| train_filtered.txt
| valid_filtered.txt
```
5. one file is corrupted. DELETE jazz.00054.wav FILE!!!
6. Run loader
```bash
make data
```

### Prepare audio for predictions

1. Convert your audio to be at most 30 seconds mono .wav file with 22050Hz sample rate
    ```bash
    # FFMPEG convertion example of mp3 file
    # we are cutting after first 50 second, next 30 seconds
    ffmpeg -i input.mp3 -ss 00:00:50 -t 00:00:30 -ar 22050 -ac 1 output.wav
    # listen to output
    ffplay output.wav
    ```
2. Move audio to `data/raw/sample_audio`

<h2 id="usage">🏋️‍♀️ Usage</h2>

### Features

You can chose the features to train SVM on in `SVM_FEATURES_FILTER` variable in `config.py`

### Models

You can download already trained models from [here](https://wutwaw-my.sharepoint.com/my?id=%2Fpersonal%2F01152433%5Fpw%5Fedu%5Fpl%2FDocuments%2FAudioConcept) and put it into `models` directory. *PROVIDED MODELS ARE TRAINED ON MPS, SO THEY WILL WORK ONLY ON MACOS SYSTEMS WITH ARM*

**We encourage to use our convenience *make* commands. For whole list run `make help`.**

### Train

Set `MODEL_TO_TRAIN` in `config.py` to 'CNN' or 'VGGish' or 'SVM' and run
```bash
python -m AudioConcept.train main
```
or set in explicitly in command
```bash
python -m AudioConcept.train main "SVM"
# or make train_{model_name}
```
The best model will be saved in model directory


If you want to have more control over chosen training parameters on the go for setting up scrips with multiple experiments use:
```bash
python -m AudioConcept.train experiment {model name} --lr {learning rate value} --audio-length {CNN for 29.1 sec or VGG for 3.96 sec} --weight-decay {weight decay value} --label-smoothing {label smoothing value} --noise-level {augmentation noise level value}
```

### Evaluate

```bash
python -m AudioConcept.evaluate
# or python -m AudioConcept.evaluate "{model_name}"
# or make evaluate_{model_name}
```
In order to see classification results for SVM you must be a member of AudioConcept project - [W&B project site](https://wandb.ai/audio-concept/audio-concept?nw=nwuserjasiostusio)

Plots will be placed in `reports/figures` directory.

<h2 id="sources">ℹ️ Sources</h2>

CNN implementation is based on [mamodrzejewski GTZAN genre classification example](https://github.com/mamodrzejewski/wimu-gtzan-genre-example), which is based on [Music Classification: Beyond Supervised Learning, Towards Real-world Applications](https://music-classification.github.io/tutorial/part3_supervised/tutorial.html) by Minz Won, Janne Spijkervet and Keunwoo Choi.

VGGish implementation is based on VGGish architecture for genre classification [paper](https://arxiv.org/pdf/1609.09430) and Aladdin Persson VGG torch [tutorial](https://www.youtube.com/watch?v=ACmuBbuXn20).

### Literature

| Title         | Comments      | Link |
| ------------- | ------------- |------|
| “Musical Genre Classification Using Advanced Audio Analysis and Deep Learning Techniques”   | Trening klasyfikacji przeprowadzono na zbiorach danych GTZAN oraz ISMIR2004. Do klasyfikacji gatunków muzycznych wykorzystano modele FNN, CNN, RNN-LSTM, SVM i KNN. Preprocessing obejmował ekstrakcję cech (MFCC, FFT, STFT), a optymalizację przeprowadzono za pomocą dropoutu, L2 regularization i batch normalization.| [link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10605044) |
| Constructing adversarial examples to investigate the plausibility of explanations in deep audio and image classifiers  | Researchers tests how plausible the explainers are by feeding them deliberately  perturbed input data. In audio domain LIME was tested and it does not handle it well. "...the explanation method LIME is not able to recover perturbed segments in a satisfactory manner, with even the baseline performing better". The tested audio was recorded voice.  | [link](https://link.springer.com/article/10.1007/s00521-022-07918-7#notes) |
| “audioLIME: Listenable Explanations Using Source Separation” | audioLIME wykorzystuje separację źródeł dźwięku, aby wyjaśnienia były słuchalne. Można stosować do modeli klasyfikujących muzykę aby zrozumieć, które komponenty dźwięku miały kluczowy wpływ na predykcję modelu.| [link](https://arxiv.org/pdf/2008.00582v3.pdf) |
| “Tracing Back Music Emotion Predictions to Sound Sources and Intuitive Perceptual Qualities” | Artykuł rozszerza audioLIME, dodając średniopoziomowe cechy percepcyjne (np. barwę, rytm, dynamikę), aby lepiej zrozumieć, jak model interpretuje emocje w muzyce. Pomaga wykryć bias w modelach klasyfikujących emocje w muzyce i sprawia, że wyjaśnienia są bliższe temu, jak ludzie rozumieją muzykę. | [link](https://arxiv.org/pdf/2106.07787v2.pdf) |
| CNN ARCHITECTURES FOR LARGE-SCALE AUDIO CLASSIFICATION | VGG-ish / Short-chunk CNNs | [link](https://arxiv.org/pdf/1609.09430) |
| MUSIC GENRE CLASSIFIER WITH DEEP NEURAL NETWORKS | CNNs with over 0.8 accuracy trained on enhanced GTZAN | [link](https://github.com/XiplusChenyu/Musical-Genre-Classification/blob/master/music_genre_classification.pdf) |
