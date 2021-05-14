'''
Main script for training and testing models
'''
import argparse
import logging
import time
import tqdm
import json
import os
import glob

def main(args):
    best_score = 0.0
    best_c1 = -1
    best_h2 = -1
    # print(glob.glob(os.path.join(args.parentdir, "*", "dev-results.json")))
    for fname in glob.glob(os.path.join(args.parentdir, "*", "dev-results.json")):
        with open(fname) as fin:
            results = json.load(fin)
            c1, h2 = fname.split('/')[-2].split('_')[-1].split('-')
            wavg = results['weighted avg']['f1-score']
            if wavg > best_score:
                best_score = wavg
                best_h1, best_h2 = c1, h2
    print(f'Best score: {best_score*100} with h1={best_h1} and h2={best_h2}')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('parentdir', help='path to save model weights or predictions')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    # logging.info(f'Time to run script: {end-start} secs')