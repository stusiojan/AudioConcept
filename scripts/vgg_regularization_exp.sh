#!/usr/bin/env sh

# python -m AudioConcept.train experiment VGGish --weight-decay 1e-4
# python -m AudioConcept.train experiment VGGish --weight-decay 1e-3

python -m AudioConcept.train experiment VGGish --weight-decay 5e-4
# python -m AudioConcept.train experiment VGGish --weight-decay 5e-3


echo "Experiments completed."