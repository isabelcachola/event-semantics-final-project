import os
import tqdm
from nltk.corpus import stopwords
import pandas as pd

def build_bow_tweet(data, tokenizer):
    ind2w = {}
    w2ind = {}
    
    w2ind["UNK"] = 0
    ind2w[0] = "UNK"

    ind = 1
    
    stop_words = set(stopwords.words('english')) 

    for doc in tqdm.tqdm(data, desc='Building BOW'):
        text = tokenizer(doc.text)
        for word in text:
            word = word.lower()
            if word not in stop_words:
                if word not in w2ind:
                    w2ind[word] = ind
                    ind2w[ind] = word
                    ind += 1
    return {
        "ind2w": ind2w,
        "w2ind": w2ind,
    }

def build_bow_reman(data, tokenizer):
    ind2w = {}
    w2ind = {}

    w2ind["UNK"] = 0
    ind2w[0] = "UNK"

    ind = 1
    stop_words = set(stopwords.words('english')) 
    for doc in tqdm.tqdm(data, desc='Building BOW'):
        text = tokenizer(doc.text)
        for word in text:
            word = word.text.lower()
            if word not in stop_words:
                if word not in w2ind:
                    w2ind[word] = ind
                    ind2w[ind] = word
                    ind += 1
    return {
        "ind2w": ind2w,
        "w2ind": w2ind,
    }
    
def get_examples_mlp(X_test, y_pred, task, outdir):
    tp, fp = [], []
    i = 0
    for doc in X_test:
        curr_tp, curr_fp = [], []
        text = doc.text
        if task == 'srl':
            y_true = doc.get_srl_labels()
        elif task == 'emotion':
            y_true = doc.get_emotion_labels()
        for (word, l_true) in y_true:
            l_hat = y_pred[i]
            if  l_true != 'O' and l_true == l_hat:
                curr_tp.append((word, l_true))
            if l_hat != 'O' and l_true != l_hat:
                curr_fp.append((word, l_true, l_hat))
            i += 1
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

def get_examples_crf(X_test, y_pred, task, outdir):
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

def print_report(report, task):
    if task == 'srl':
        labels = [ 'target',
                'experiencer',
                'emotion',
                'macro avg',
                'micro avg',
                'weighted avg']
    else:
        labels = [
            'anger',
            'fear',
            'trust',
            'disgust',
            'joy',
            'sadness',
            'surprise',
            'anticipation',
            'macro avg',
            'micro avg',
            'weighted avg',
        ]
    print('precision\trecall\tf1-score')
    rows = []
    for l in labels:
        row = [l]
        line = l + '\t'
        for metric in ['precision', 'recall', 'f1-score']:
            score = round(report[l][metric]*100, 1)
            line += str(score) + '\t'
            row.append(score)
        print(line)
        rows.append(row)
    df = pd.DataFrame(rows, columns=['label', 'precision', 'recall','f1'])
    return df