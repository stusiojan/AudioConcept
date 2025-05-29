#!/usr/bin/env sh
echo "Running CNN experiments for AudioConcept..."

# learning rate experiments for CNN model
python -m AudioConcept.train experiment CNN --lr 1e-3
python -m AudioConcept.train experiment CNN --lr 5e-4
python -m AudioConcept.train experiment CNN --lr 2e-3

# TODO: change lr for the best one
# reduced regularization experiments for CNN model
python -m AudioConcept.train experiment CNN --lr 1e-3 --weight-decay 1e-6 --label-smoothing 0.0
python -m AudioConcept.train experiment CNN --lr 1e-3 --weight-decay 1e-6 --label-smoothing 0.2

# simple augmentation impact (noise)
python -m AudioConcept.train experiment CNN --lr 1e-3 --noise-level 0.0
python -m AudioConcept.train experiment CNN --lr 1e-3 --noise-level 0.005

echo "Experiments completed."