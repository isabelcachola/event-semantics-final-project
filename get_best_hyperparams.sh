#!/bin/bash
for DATASET in reman tweet; do
    for TASK in srl emotion; do
        echo $DATASET $TASK;
        OUTDIR="model_weights/crf/$DATASET-$TASK/";
        python scripts/get_best_hyperparams.py $OUTDIR;
    done
done

for DATASET in reman tweet; do
    for TASK in srl emotion; do
        echo $DATASET $TASK;
        OUTDIR="model_weights/mlp/$DATASET-$TASK/";
        python scripts/get_best_hyperparams.py $OUTDIR;
    done
done
