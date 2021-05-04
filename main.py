'''
Main script for training and testing models
'''
import argparse
import logging
import time

def read_data(dataset, datadir):
    if dataset == 'reman':
        NotImplementedError()

    elif dataset == 'tweet':
        NotImplementedError()
    else:
        raise ValueError(f'Uknown dataset type {dataset}')

def init_model(model_type):
    if model_type =='logreg':
        NotImplementedError()
    elif model_type == 'crf':
        NotImplementedError()
    elif model_type == 'mlp':
        NotImplementedError()
    else:
        raise ValueError(f'Uknown model type {model_type}')

def train(model, data):
    NotImplementedError()

def test(model, data):
    NotImplementedError()

def main(args):
    data = read_data(args.dataset, args.datadir)

    model = init_model(args.model_type)

    if args.mode == 'train':
        model = train(model, data, args.outdir)
    elif args.mode == 'test':
        test(model, data)
    else:
        ValueError(f'Unkown mode {args.mode}')

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