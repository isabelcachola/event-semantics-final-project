'''
Main script for training and testing models
'''
import argparse
import logging
import time
import tqdm
import json
import os
import pickle
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from nltk.tokenize import TweetTokenizer

from data import RemanData, TweetData
from utils import build_bow_reman, build_bow_tweet, get_examples_crf, get_examples_mlp
from models import CRF, MLP

def read_data(dataset, datadir, split='train'):
    if dataset == 'reman':
        data = RemanData(datadir, split)
        data.read_data()

        # Either load bow or build bow
        dict_path = os.path.join(datadir, 'reman-vocab.json')
        if os.path.exists(dict_path) or split !='train':
            dicts = json.load(open(dict_path))
        else:
            nlp = English()
            tokenizer = Tokenizer(nlp.vocab)
            dicts = build_bow_reman(data, tokenizer)
            with open(dict_path, 'w') as fout:
                json.dump(dicts, fout, indent=4)

    elif dataset == 'tweet':
        data = TweetData(datadir, split)
        data.read_data()

        dict_path = os.path.join(datadir, 'tweet-vocab.json')
        if os.path.exists(dict_path)  or split !='train':
            dicts = json.load(open(dict_path))
        else:
            tokenizer = TweetTokenizer().tokenize
            dicts = build_bow_tweet(data, tokenizer)
            with open(dict_path, 'w') as fout:
                json.dump(dicts, fout, indent=4)

    else:
        raise ValueError(f'Uknown dataset type {dataset}')

    logging.info(f'Vocab size: {len(dicts["ind2w"])}')

    return data, dicts

def init_model(model_type, task, dataset,
                c1=0.1, c2=0.1,
                nepochs=5, 
                batch_size=32,
                hidden_dim=512):
    if model_type == 'crf':
        model = CRF(task, c1=c1, c2=c2)
    elif model_type == 'mlp':
        model = MLP(task, dataset, 
                nepochs=nepochs, 
                batch_size=batch_size,
                hidden_dim=hidden_dim)
    else:
        raise ValueError(f'Uknown model type {model_type}')

    return model

def train(model, train_data, dev_data, outdir):
    X_train, y_train = model.featurize_data(train_data)
    model.train(X_train, y_train)

    X_dev, y_dev = model.featurize_data(dev_data)
    dev_results, _ = model.test(X_dev, y_dev)
    # dev_results.to_csv(os.path.join(outdir, 'dev-results.csv'), index=False)
    with open(os.path.join(outdir, 'dev-results.json'), 'w') as fout:
        json.dump(dev_results, fout)

    return model

def test(model, data, task, model_type, outdir):
    X_test, y_test = model.featurize_data(data)
    results, y_pred = model.test(X_test, y_test)
    if model_type == 'crf':
        get_examples_crf(data, y_pred, task, outdir)
    else:
        get_examples_mlp(data, y_pred, task, outdir)
    with open(os.path.join(outdir, 'test-results.json'), 'w') as fout:
        json.dump(results, fout)
    # results.to_csv(os.path.join(outdir, 'test-results.csv'), index=False)

def main(args):
    model = init_model(args.model_type, args.task, args.dataset, 
                        c1=args.c1, c2=args.c2,
                        nepochs=args.nepochs, 
                        batch_size=args.batch_size,
                        hidden_dim=args.hidden_dim)
    if args.model_type == 'crf':
        model_path = os.path.join(args.outdir, f'{args.dataset}-{args.task}-{args.model_type}.joblib')
    else:
        model_path = args.outdir

    if args.mode == 'train':
        train_data, dicts = read_data(args.dataset, args.datadir, split='train')
        dev_data, _ = read_data(args.dataset, args.datadir, split='dev')
        model = train(model, train_data, dev_data, args.outdir)
        model.save(model_path)

    elif args.mode == 'test':
        model.load(model_path)
        test_data, dicts = read_data(args.dataset, args.datadir, split='test')
        test(model, test_data, args.task, args.model_type, args.outdir)

    else:
        ValueError(f'Unkown mode {args.mode}')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train','test'])
    parser.add_argument('model_type', choices=['crf','mlp'])
    parser.add_argument('dataset', choices=['reman','tweet'])
    parser.add_argument('task', choices=['srl','emotion'])
    parser.add_argument('outdir', help='path to save model weights or predictions')
    parser.add_argument('--datadir', default='./data', help='path to data dir')
    # CRF Hyperparams
    parser.add_argument('--c1', default=0.1, type=float)
    parser.add_argument('--c2', default=0.1, type=float)
    # MLP Hyperparams
    parser.add_argument('--nepochs', default=5, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')