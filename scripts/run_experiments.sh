#!/usr/bin/env sh
echo "Running CNN experiments for AudioConcept..."

# learning rate experiments for CNN model
# python -m AudioConcept.train experiment CNN --lr 1e-3
# python -m AudioConcept.train experiment CNN --lr 5e-4
# python -m AudioConcept.train experiment CNN --lr 2e-3

# # TODO: change lr for the best one
# # reduced regularization experiments for CNN model
# python -m AudioConcept.train experiment CNN --lr 1e-3 --weight-decay 1e-6 --label-smoothing 0.0
# python -m AudioConcept.train experiment CNN --lr 1e-3 --weight-decay 1e-6 --label-smoothing 0.2

# # simple augmentation impact (noise)
# python -m AudioConcept.train experiment CNN --lr 1e-3 --noise-level 0.0
# python -m AudioConcept.train experiment CNN --lr 1e-3 --noise-level 0.005


# python -m AudioConcept.train experiment VGGish --audio-length VGG --lr 5e-5
# python -m AudioConcept.train experiment VGGish --audio-length VGG --lr 6e-5
# python -m AudioConcept.train experiment VGGish --audio-length VGG --lr 8e-5


# python -m AudioConcept.train experiment VGGish --audio-length VGG --lr 8e-6
# python -m AudioConcept.train experiment VGGish --audio-length VGG --lr 1e-5
# python -m AudioConcept.train experiment VGGish --audio-length VGG --lr 3e-5

# second round
# python -m AudioConcept.train experiment VGGish --audio-length VGG --lr 4e-5 --weight-decay 8e-6
python -m AudioConcept.train experiment VGGish --audio-length VGG --lr 4e-5 --weight-decay 5e-5
# python -m AudioConcept.train experiment VGGish --audio-length VGG --lr 4e-5 --weight-decay 1e-4
# python -m AudioConcept.train experiment VGGish --audio-length VGG --lr 4e-5 --weight-decay 8e-4
# python -m AudioConcept.train experiment VGGish --audio-length VGG --lr 4e-5 --weight-decay 1e-3
# python -m AudioConcept.train experiment VGGish --audio-length VGG --lr 4e-5 --weight-decay 1e-2


# python -m AudioConcept.train experiment CNN --audio-length CNN --lr 8e-2 --weight-decay 5e-4

# python -m AudioConcept.train experiment CNN --audio-length CNN --lr 1e-3 --weight-decay 5e-4
# python -m AudioConcept.train experiment CNN --audio-length CNN --lr 5e-3 --weight-decay 5e-4
# python -m AudioConcept.train experiment CNN --audio-length CNN --lr 7e-3 --weight-decay 5e-4
# python -m AudioConcept.train experiment CNN --audio-length CNN --lr 9e-3 --weight-decay 5e-4

# python -m AudioConcept.train experiment CNN --audio-length CNN --lr 5e-4 --weight-decay 5e-4

# python -m AudioConcept.train experiment CNN --audio-length CNN --lr 9e-4 --weight-decay 5e-4

echo "Experiments completed."