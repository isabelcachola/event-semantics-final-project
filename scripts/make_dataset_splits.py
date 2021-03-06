'''
Sprict to make dataset splits
'''
import argparse
import logging
import time
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
import random
from data import TweetData
random.seed()


def split_tweets(datadir):
    tweet_data = TweetData(datadir, None)
    tweet_data.read_data()
    doc_ids = set()
    for doc in tweet_data:
        doc_ids.add(doc.metadata["unitid"])
    
    doc_ids = list(doc_ids)
    random.shuffle(doc_ids)
    
    split1 = int(len(doc_ids)*0.7)
    split2 = int(len(doc_ids)*0.85)

    train, dev, test = doc_ids[:split1], doc_ids[split1:split2], doc_ids[split2:]

    outdir = os.path.join(datadir, "ElectoralTweetsData/Annotated-US2012-Election-Tweets/Questionnaire2")

    train_file = open(os.path.join(outdir, 'train_ids.txt'), 'w')
    dev_file = open(os.path.join(outdir, 'dev_ids.txt'), 'w')
    test_file = open(os.path.join(outdir, 'test_ids.txt'), 'w')

    for doc in tweet_data:
        did = doc.metadata["unitid"]
        if did in train:
            train_file.write(did + '\n')
        elif did in dev:
            dev_file.write(did + '\n')
        elif did in test:
            test_file.write(did + '\n')
        else:
            raise Exception

    train_file.close()
    dev_file.close()
    test_file.close()

def split_reman(datadir):
    tree = ET.parse(os.path.join(datadir, 'reman', 'reman-version1.0.xml'))
    root = tree.getroot()
    doc_ids = set()
    for doc in tqdm(root):
        try:
            did, _ = doc.attrib['doc_id'].split('|')
        except ValueError:
            did = doc.attrib['doc_id']
        doc_ids.add(did)

    doc_ids = list(doc_ids)
    random.shuffle(doc_ids)

    split1 = int(len(doc_ids)*0.7)
    split2 = int(len(doc_ids)*0.85)

    train, dev, test = doc_ids[:split1], doc_ids[split1:split2], doc_ids[split2:]

    train_file = open(os.path.join(datadir, 'reman', 'train_ids.txt'), 'w')
    dev_file = open(os.path.join(datadir, 'reman', 'dev_ids.txt'), 'w')
    test_file = open(os.path.join(datadir, 'reman', 'test_ids.txt'), 'w')
    for doc in root:
        try:
            did, _ = doc.attrib['doc_id'].split('|')
        except ValueError:
            did = doc.attrib['doc_id']
        if did in train:
            train_file.write(doc.attrib['doc_id'] + '\n')
        elif did in dev:
            dev_file.write(doc.attrib['doc_id'] + '\n')
        elif did in test:
            test_file.write(doc.attrib['doc_id'] + '\n')
        else:
            raise Exception

    train_file.close()
    dev_file.close()
    test_file.close()

def main(args):
    if args.dataset == 'reman':
        split_reman(args.datadir)
    elif args.dataset == 'tweet':
        split_tweets(args.datadir)
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['reman','tweet'])
    # parser.add_argument('outdir', help='path to save model weights or predictions')
    parser.add_argument('--datadir', default='../data', help='path to data dir')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')