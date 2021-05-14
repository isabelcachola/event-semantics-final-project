#!/bin/bash
mkdir model_weights/crf;
for TASK in srl emotion; do
    for DATASET in tweet reman; do
        mkdir model_weights/crf/$DATASET-$TASK;
        for c1 in 0.001 0.01 0.03 0.05 0.1; do
            for c2 in 0.001 0.01 0.03 0.05 0.1; do
                OUTDIR="model_weights/crf/$DATASET-$TASK/crf_$c1-$c2";
                mkdir $OUTDIR;
                python main.py train crf $DATASET $TASK $OUTDIR --c1 $c1 --c2 $c2;
            done
        done
    done
done

mkdir model_weights/mlp;
for TASK in srl emotion; do
    for DATASET in tweet reman; do
        mkdir model_weights/mlp/$DATASET-$TASK;
        for bsz in 8 32 64; do
            for hidden_dim in 256 512; do
                OUTDIR="model_weights/mlp/$DATASET-$TASK/mlp_$bsz-$hidden_dim";
                mkdir $OUTDIR;
                python main.py train mlp $DATASET $TASK $OUTDIR --batch_size $bsz --hidden_dim $hidden_dim --nepochs 10;
            done
        done
    done
done