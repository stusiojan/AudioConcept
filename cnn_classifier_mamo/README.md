# GZTAN genre CNN classificator

Based on [mamodrzejewski GTZAN genre classification example](https://github.com/mamodrzejewski/wimu-gtzan-genre-example), which is based on [Music Classification: Beyond Supervised Learning, Towards Real-world Applications](https://music-classification.github.io/tutorial/part3_supervised/tutorial.html) by Minz Won, Janne Spijkervet and Keunwoo Choi.

I am using a machine with apple sillicon, so the code will be modified to run well on ARM architecture.

In this project we are using wandb for experiments monitoring, so this will be changed from tensorboard.

# Run

## Prepare conda environment

Install environment (from `environment.yml` on linux, `environment_mac.yml` on macOS)
```bash
conda env create -f environment_mac.yml
conda activate wimu
```

## Prepare data

1. Download GTZAN dataset and set path to it in `config.py`
2. Run get data in GTZAN dataset repository
```bash
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

## Train

1. Set `MODEL_TO_TRAIN` in `config.py` to 'CNN' or 'VGGish'
2. Create a directory `models` in `cnn_classifier_mamo` directory if not exist
3.
```bash
python train.py
```
The best model will be saved in model directory

## Evaluate

```bash
python evaluate.py
```

The confustion matrix will be saved in results directory
