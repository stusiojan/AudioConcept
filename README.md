# AudioConcept

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Locating Musical Concepts in a Genre Classifier.

## SETUP

#### Prepare virtual environment and download dependencies
```bash
python 3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

--------

Log in to Weights and biases
```bash
wandb login
# paste your API key from website https://wandb.ai/home
```

## USAGE

To train and plot SVM results use commands:
```bash
python -m AudioConcept.modeling.train
```

In order to see classification results you must be a member of AudioConcept project - [W&B project site](https://wandb.ai/audio-concept/audio-concept?nw=nwuserjasiostusio)

Plots will be placed in `reports/figures` directory.

