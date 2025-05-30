# AudioConcept

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Locating Musical Concepts in a Genre Classifier.

## SETUP

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

### Log into WandB

Log in to Weights and biases
```bash
wandb login
# paste your API key from website https://wandb.ai/home
```
### Prepare data

1. Download GTZAN dataset and set path to it in `config.py`
2. Run get data in GTZAN dataset repository
```bash
cd scripts
./get_data.sh
python gtzan_loader.py
```
3. Make sure that folder names are correct
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
4. DELETE jazz.00054.wav FILE!!!
5. Run loader
```bash
python -m AudioConcept.dataset
```

## USAGE
### Train

Set `MODEL_TO_TRAIN` in `config.py` to 'CNN' or 'VGGish' or 'SVM'
```bash
python -m AudioConcept.train main
```
or chose it each time (but there could be mistakes in model name in logs)
```bash
python -m AudioConcept.train main "SVM"
```

The best model will be saved in model directory

If you want to have more control over training parameters on the go for setting up scrips with multiple experiments use:
```bash
python -m AudioConcept.train experiment {model name} --lr {learning rate value} --weight-decay {weight decay value} --label-smoothing {label smoothing value}  --noise-level {augmentation noise level value}
```

If you want to train VGGish on 3.96 second long audio change train_loader, valid_loader and test_loader `audio_length` parameter to `AudioLength.VGG`

### Evaluate

```bash
python -m AudioConcept.evaluate
# python -m AudioConcept.evaluate "SVM"
```
In order to see classification results you must be a member of AudioConcept project - [W&B project site](https://wandb.ai/audio-concept/audio-concept?nw=nwuserjasiostusio)

Plots will be placed in `reports/figures` directory.

### Predict genre

1. Add an audio file to `data/raw/sample_audio`. Remember that audio should has .wav extension, has 22050 Hz, be no longer than 60 seconds and has only one channel.
2. Run:
    ```bash
    python -m AudioConcept.predict CNN test.wav
    ```

    change *CNN* to any model from VGGish, CNN and SVM.

    change *test.wav* to your filename.

# Sources

CNN implementation is based on [mamodrzejewski GTZAN genre classification example](https://github.com/mamodrzejewski/wimu-gtzan-genre-example), which is based on [Music Classification: Beyond Supervised Learning, Towards Real-world Applications](https://music-classification.github.io/tutorial/part3_supervised/tutorial.html) by Minz Won, Janne Spijkervet and Keunwoo Choi.

VGGish implementation is based on VGGish architecture for genre classification [paper](https://arxiv.org/pdf/1609.09430) and Aladdin Persson VGG torch [tutorial](https://www.youtube.com/watch?v=ACmuBbuXn20).