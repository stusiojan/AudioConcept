DATA_PATH = "../data/raw/gtzan"
LEARNING_RATE = 3e-4
NUM_EPOCHS = 30
VGG16_ARCHITECTURE = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
    # Then flatten
    # Then 4096x4096x1000 linear layers - here 4096x4096x10
]
MODEL_TO_TRAIN = "VGGish"  # "VGGish" or "CNN"
