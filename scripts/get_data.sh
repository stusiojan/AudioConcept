#!/usr/bin/env sh
# wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
# tar -zxvf genres.tar.gz
cd ../data/raw/gtzan
wget https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/train_filtered.txt
wget https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/valid_filtered.txt
wget https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/test_filtered.txt