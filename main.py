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
from models import LogisticRegression, CRF, MLP
from utils import build_bow_reman, build_bow_tweet

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

def init_model(model_type, task, c1=0.1, c2=0.1):
    if model_type == 'crf':
        model = CRF(task, c1=c1, c2=c2)
    elif model_type == 'mlp':
        model = MLP()
    else:
        raise ValueError(f'Uknown model type {model_type}')

    return model

def train(model, train_data, dev_data, outdir):
    X_train, y_train = model.featurize_data(train_data)
    model.train(X_train, y_train)

    X_dev, y_dev = model.featurize_data(dev_data)
    dev_results, _ = model.test(X_dev, y_dev)
    with open(os.path.join(outdir, 'dev-results.json'), 'w') as fout:
        json.dump(dev_results, fout)

    return model

def get_examples(X_test, y_pred, task, outdir):
    tp, fp = [], []
    for doc, y_hat in zip(X_test, y_pred):
        curr_tp, curr_fp = [], []
        text = doc.text
        if task == 'srl':
            y_true = doc.get_srl_labels()
        elif task == 'emotion':
            y_true = doc.get_emotion_labels()
        for (word, l_true), l_hat in zip(y_true, y_hat):
            if  l_true != 'O' and l_true == l_hat:
                curr_tp.append((word, l_true))
            if l_hat != 'O' and l_true != l_hat:
                curr_fp.append((word, l_true, l_hat))
        tp.append([text, curr_tp])
        fp.append([text,curr_fp])
    
    with open(os.path.join(outdir, 'tp.txt'), 'w') as fout:
        for doc in tp:
            if len(doc[1]) > 0:
                fout.write(f'{doc[0]}\n')
                for w in doc[1]:
                    fout.write(f'Word:{w[0]}\tYTrue:{w[1]}\n')
                fout.write('\n\n')

    with open(os.path.join(outdir, 'fp.txt'), 'w') as fout:
        for doc in fp:
            if len(doc[1]) > 0:
                fout.write(f'{doc[0]}\n')
                for w in doc[1]:
                    fout.write(f'Word:{w[0]}\tYTrue:{w[1]}\tYPred:{w[2]}\n')
                fout.write('\n\n')

def test(model, data, task, outdir):
    X_test, y_test = model.featurize_data(data)
    results, y_pred = model.test(X_test, y_test)
    get_examples(data, y_test, y_pred, task, outdir)
    with open(os.path.join(outdir, 'test-results.json'), 'w') as fout:
        json.dump(results, fout)

def main(args):
    model = init_model(args.model_type, args.task, c1=args.c1, c2=args.c2)
    model_path = os.path.join(args.outdir, f'{args.dataset}-{args.task}-{args.model_type}.joblib')

    if args.mode == 'train':
        train_data, dicts = read_data(args.dataset, args.datadir, split='train')
        dev_data, _ = read_data(args.dataset, args.datadir, split='dev')
        model = train(model, train_data, dev_data, args.outdir)
        model.save(model_path)

    elif args.mode == 'test':
        model.load(model_path)
        test_data, dicts = read_data(args.dataset, args.datadir, split='test')
        test(model, test_data, args.task, args.outdir)

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
    parser.add_argument('--c1', default=0.1, type=float)
    parser.add_argument('--c2', default=0.1, type=float)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')