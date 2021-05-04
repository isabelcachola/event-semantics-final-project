import os
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import scipy
import nltk
import joblib

class LogisticRegression:
    def __init__(self):
        pass

class CRF:
    def __init__(self, task):
        self.task = task
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

    def featurize_data(self, data):
        X = []
        y =[]
        
        for i, doc in enumerate(data):
            if self.task == 'srl':
                sents = doc.get_srl_labels()
            elif self.task == 'emotion':
                sents = doc.get_emotion_labels()
            X.append(self.sent2features(sents))
            y.append(self.sent2labels(sents))

        return X, y

    def sent2labels(self, sent):
        return [word[1] for word in sent]

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def word2features(self, doc, i):
        word = doc[i][0]

        # Common features for all words. You may add more features here based on your custom use case
        features = [
                'bias',
                'word.lower=' + word.lower(),
                'word[-3:]=' + word[-3:],
                'word[-2:]=' + word[-2:],
                'word.isupper=%s' % word.isupper(),
                'word.isdigit=%s' % word.isdigit()
            ]

        # Features for words that are not at the beginning of a document
        if i > 0:
                word1 = doc[i-1][0]
                features.extend([
                    '-1:word.lower=' + word1.lower(),
                    '-1:word.istitle=%s' % word1.istitle(),
                    '-1:word.isupper=%s' % word1.isupper(),
                    '-1:word.isdigit=%s' % word1.isdigit()
                ])
        else:
            # Indicate that it is the 'beginning of a document'
            features.append('BOS')

        # Features for words that are not at the end of a document
        if i < len(doc)-1:
                word1 = doc[i+1][0]
                features.extend([
                    '+1:word.lower=' + word1.lower(),
                    '+1:word.isupper=%s' % word1.isupper(),
                    '+1:word.isdigit=%s' % word1.isdigit()
                ])
        else:
            # Indicate that it is the 'end of a document'
            features.append('EOS')

        return features

    def train(self, X, y):
        self.crf.fit(X, y)

    def test(self, X, y):
        y_pred = self.crf.predict(X)
        labels = list(self.crf.classes_)
        labels.remove('O')
        print(metrics.flat_classification_report(y, y_pred,
                            labels=labels))

    def save(self, outpath):
        joblib.dump(self.crf, outpath)
    
    def load(self, outpath):
        self.crf = joblib.load(outpath)
    
class MLP:
    def __init__(self):
        pass