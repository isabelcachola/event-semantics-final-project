#!/bin/bash
TASK=emotion
DATASET=tweet

mkdir model_weights/crf
mkdir model_weights/crf/$DATASET-$TASK
for c1 in 0.001 0.01 0.03 0.05 0.1; do
    for c2 in 0.001 0.01 0.03 0.05 0.1; do
        OUTDIR="model_weights/crf/$DATASET-$TASK/crf_$c1-$c2";
        echo "training $OUTDIR";
        mkdir $OUTDIR;
        python main.py train crf $DATASET $TASK $OUTDIR --c1 $c1 --c2 $c2
    done
done