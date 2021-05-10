import os
import scipy
import nltk
import numpy as np
import joblib
import pickle
from pprint import pprint
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier

class CRF:
    def __init__(self, task, c1=0.1, c2=0.1, max_iterations=100,):
        self.task = task
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
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
        report  = metrics.flat_classification_report(y, y_pred, labels=labels, output_dict=True)
        pprint(report)
        return report, y_pred

    def save(self, outpath):
        joblib.dump(self.crf, outpath)
    
    def load(self, outpath):
        self.crf = joblib.load(outpath)
    
class MLP:
    def __init__(self, task, dataset,
                nepochs=1,
                hidden_dim=512, 
                batch_size=32):
        self.task = task
        self.dataset = dataset
        self.dict_vectorizer = DictVectorizer(sparse=False)
        self.label_encoder = LabelEncoder()
        
        #hyper params
        # self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # self.output_dim = 4 if task=='srl' else 9
        self.batch_size = batch_size

        self.model_params = {
            'build_fn': self.build_model,
            'epochs': nepochs,
            'batch_size': batch_size,
            'verbose': 1,
            'shuffle': True
        }
        self.clf = KerasClassifier

    def build_model(self):
        model = Sequential([
            Dense(self.hidden_dim, input_dim=self.input_dim),
            Activation('relu'),
            Dropout(0.2),
            Dense(self.hidden_dim),
            Activation('relu'),
            Dropout(0.2),
            Dense(self.output_dim, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        return model

    def add_basic_features(self, sentence_terms, index):
        """ Compute some very basic word features.
            :param sentence_terms: [w1, w2, ...] 
            :type sentence_terms: list
            :param index: the index of the word 
            :type index: int
            :return: dict containing features
            :rtype: dict
        """
        term = sentence_terms[index]
        return {
            'nb_terms': len(sentence_terms),
            'term': term,
            'is_first': index == 0,
            'is_last': index == len(sentence_terms) - 1,
            'is_capitalized': term[0].upper() == term[0],
            'is_all_caps': term.upper() == term,
            'is_all_lower': term.lower() == term,
            'prefix-1': term[0],
            'prefix-2': term[:2],
            'prefix-3': term[:3],
            'suffix-1': term[-1],
            'suffix-2': term[-2:],
            'suffix-3': term[-3:],
            'prev_word': '' if index == 0 else sentence_terms[index - 1],
            'next_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 1]
        }

    def untag(self, tagged_sentence):
        """ 
            Remove the tag for each tagged term.
        :param tagged_sentence: a POS tagged sentence
            :type tagged_sentence: list
            :return: a list of tags
            :rtype: list of strings
            """
        return [w for w, _ in tagged_sentence]
    def featurize_data(self, dataset):
        """
        Split tagged sentences to X and y datasets and append some basic features.
        :param tagged_sentences: a list of POS tagged sentences
            :param tagged_sentences: list of list of tuples (term_i, tag_i)
            :return: 
            """
        X, y = [], []
        for doc in dataset:
            labeled_data = doc.get_srl_labels() if self.task =='srl' else doc.get_emotion_labels()
            for index, (term, class_) in enumerate(labeled_data):
                # Add basic NLP features for each sentence term
                X.append(self.add_basic_features(self.untag(labeled_data), index))
                y.append(class_)
        return X, y
    
    def plot_model_performance(self, train_loss, train_acc, train_val_loss, train_val_acc):
        """ Plot model loss and accuracy through epochs. """
        blue= '#34495E'
        green = '#2ECC71'
        orange = '#E23B13'
        # plot model loss
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
        ax1.plot(range(1, len(train_loss) + 1), train_loss, blue, linewidth=5, label='training')
        ax1.plot(range(1, len(train_val_loss) + 1), train_val_loss, green, linewidth=5, label='validation')
        ax1.set_xlabel('# epoch')
        ax1.set_ylabel('loss')
        ax1.tick_params('y')
        ax1.legend(loc='upper right', shadow=False)
        ax1.set_title('Model loss through #epochs', color=orange, fontweight='bold')
        # plot model accuracy
        ax2.plot(range(1, len(train_acc) + 1), train_acc, blue, linewidth=5, label='training')
        ax2.plot(range(1, len(train_val_acc) + 1), train_val_acc, green, linewidth=5, label='validation')
        ax2.set_xlabel('# epoch')
        ax2.set_ylabel('accuracy')
        ax2.tick_params('y')
        ax2.legend(loc='lower right', shadow=False)
        ax2.set_title('Model accuracy through #epochs', color=orange, fontweight='bold')

    def train(self, X_train, y_train, X_dev=None, y_dev=None):
        self.dict_vectorizer.fit(X_train)
        self.label_encoder.fit(y_train)

        X_train = self.dict_vectorizer.transform(X_train)
        y_train =  np_utils.to_categorical(self.label_encoder.transform(y_train))

        print(X_train.shape[1])
        self.input_dim = X_train.shape[1]
        self.output_dim = y_train.shape[1]

        if X_dev is not None and y_dev is not None:
            X_dev = self.dict_vectorizer.transform(X_dev)
            y_dev =  np_utils.to_categorical(self.label_encoder.transform(y_dev))
            self.model_params['validation_data'] = (X_dev, y_dev),

        self.clf = self.clf(**self.model_params)
        hist = self.clf.fit(X_train, y_train)
        self.plot_model_performance(
            train_loss=hist.history.get('loss', []),
            train_acc=hist.history.get('acc', []),
            train_val_loss=hist.history.get('val_loss', []),
            train_val_acc=hist.history.get('val_acc', [])
        )
    
    def test(self, X_test, y_test):
        X_test = self.dict_vectorizer.transform(X_test)
        y_pred = np.argmax(self.clf.predict(X_test), axis=1)
        y_pred = self.label_encoder.inverse_transform(y_pred).tolist()
        labels =  self.label_encoder.classes_.tolist()
        labels.remove('O')
        report  = classification_report(y_test, y_pred, labels=labels, output_dict=True)
        pprint(report)
        return report, y_pred

    def save(self, outdir):
        self.clf.model.save(os.path.join(outdir, f'{self.dataset}-{self.task}-mlp.keras'))
        pickle.dump(self.dict_vectorizer, open(os.path.join(outdir, f'{self.dataset}-{self.task}-vectorizer.pickle'), "wb"))
        pickle.dump(self.label_encoder, open(os.path.join(outdir, f'{self.dataset}-{self.task}-labelencoder.pickle'), "wb"))

    
    def load(self, outdir):
        self.clf = tf.keras.models.load_model(os.path.join(outdir, f'{self.dataset}-{self.task}-mlp.keras'))
        self.dict_vectorizer = pickle.load(open(os.path.join(outdir, f'{self.dataset}-{self.task}-vectorizer.pickle'), "rb"))
        self.label_encoder = pickle.load(open(os.path.join(outdir, f'{self.dataset}-{self.task}-labelencoder.pickle'), "rb"))


