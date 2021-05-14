'''
Sprict to make dataset splits
'''
import argparse
import logging
import time
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from data import TweetData, RemanData
from wordcloud import WordCloud, STOPWORDS
from PIL import Image


def create_word_cloud(string, outfile):
   cloud = WordCloud(background_color = "white", max_words = 200, stopwords = set(STOPWORDS))
   cloud.generate(string)
   cloud.to_file(outfile)

def get_stats(dataset):
    # Num documents
    print(f'Num docs: {len(dataset)}')

    # Avg len of example and unique tokens
    len_examples = []
    unique_tokens = set()
    big_string = ''
    for doc in tqdm(dataset):
        example = doc.get_srl_labels()
        big_string += ' ' + doc.text
        len_examples.append(len(example))
        for word, _ in example:
            unique_tokens.add(word)
    print(f'Num unique tokens: {len(unique_tokens)}')
    print(f'Mean len ex: {np.mean(len_examples)}')
    print(f'Mean len ex: {np.std(len_examples)}')

    return len_examples, big_string


def main(args):
    reman = RemanData(args.datadir, None)
    reman.read_data()

    tweet = TweetData(args.datadir, None)
    tweet.read_data()

    len_examples_reman, reman_str = get_stats(reman)
    len_examples_tweet, tweet_str = get_stats(tweet)

    create_word_cloud(reman_str, 'reman_wordcloud.png')
    create_word_cloud(tweet_str, 'tweet_wordcloud.png')

    data = {
        'Twitter': len_examples_tweet,
        'Reman': len_examples_reman
    }

    fig, ax = plt.subplots()
    bp = ax.boxplot(data.values(), patch_artist=True)
    FONTSIZE = 20
    for patch in bp['boxes']:
        patch.set(facecolor='lightblue', edgecolor='black') 
    for patch in bp['medians']:
        patch.set(color='black') 
    for patch in bp['fliers']:
        patch.set(markerfacecolor='black', markersize=6, markeredgecolor='none') 
    ax.set_xticklabels(data.keys(), fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    ax.set_ylabel('Number of tokens', fontsize=FONTSIZE)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('dataset', choices=['reman','tweet'])
    # parser.add_argument('outdir', help='path to save model weights or predictions')
    parser.add_argument('--datadir', default='../data', help='path to data dir')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')