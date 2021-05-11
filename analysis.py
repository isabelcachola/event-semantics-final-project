import argparse
import logging
import time
import pandas as pd
from data import RemanData, TweetData
import matplotlib.pyplot as plt


def read_data(dataset, datadir):
    if dataset == 'reman':
        reman_data = RemanData(datadir, None)
        reman_data.read_data()
        return reman_data
    elif dataset == 'tweet':
        tweet_data = TweetData(datadir, None)
        tweet_data.read_data()
        return tweet_data
    else:
        raise ValueError(f'Uknown dataset type {dataset}')


def emotion_dist(data):
    dataset_length = len(data)
    emotions = {}

    for i in range(dataset_length):
        em = data[i].emotions
        if len(em) != 0:
            print(em)
            for e in em:
                e = e.label
                if e in emotions:
                    emotions[e] += 1
                else:
                    emotions[e] = 1
                print(e)

    for v in emotions.keys():
        emotions[v] /= dataset_length
    print(emotions)

    sorted_em = dict(sorted(emotions.items(), key=lambda item: item[1], reverse=True))

    plt.figure(figsize=(8, 5))
    plt.bar(*zip(*sorted_em.items()))
    plt.xlabel("Emotions")
    plt.ylabel("Percentage")
    plt.title("Reman Dataset")
    plt.show()
    print(dataset_length)

def srl_dist(data):
    dataset_length = len(data)
    srls = {}

    for i in range(dataset_length):
        em = data[i].srls
        if len(em) != 0:
            print(em)
            for e in em:
                e = e.label
                if e in srls:
                    srls[e] += 1
                else:
                    srls[e] = 1
                print(e)

    for v in srls.keys():
        srls[v] /= dataset_length
    print(srls)

    sorted_em = dict(sorted(srls.items(), key=lambda item: item[1], reverse=True))

    plt.figure(figsize=(8, 5))
    plt.bar(*zip(*sorted_em.items()))
    plt.xlabel("Semantic Role")
    plt.ylabel("Percentage")
    plt.title("Twitter Dataset")
    plt.show()
    print(dataset_length)


def main(args):
    data = read_data(args.dataset, args.datadir)
    srl_dist(data)
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train','test'])
    parser.add_argument('model_type', choices=['logreg','crf','mlp'])
    parser.add_argument('dataset', choices=['reman','tweet'])
    parser.add_argument('outdir', help='path to save model weights or predictions')
    parser.add_argument('--datadir', default='./data', help='path to data dir')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')