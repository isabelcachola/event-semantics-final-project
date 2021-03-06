import xml.etree.ElementTree as ET
import os
import tqdm
import pandas as pd
import nltk
import textspan
from mapping import REMAN_SRL_MAPPING, TWEET_EMOTION_MAPPING, REMAN_EMOTION_MAPPING

class Span:
    def __init__(self, cbegin, cend, label, type):
        if int(cbegin) < int(cend):
            self.cbegin = int(cbegin)
            self.cend = int(cend)
        else:
            self.cbegin = int(cend)
            self.cend = int(cbegin)
        self.type = type # Options are "emotion" or "srl"
        self.label = label # Actual emotion or semantic role label

    def __repr__(self):
        return f'{self.type}: {self.label} ({self.cbegin}:{self.cend})'

class Document:
    def __init__(self, metadata, text, emotions, srls):

        self.metadata = metadata # Metadata provided by datasets, not used during training
        
        self.text = text
        self.emotions = emotions
        self.srls = srls
    
    def __repr__(self):
        return self.text

    def get_srl_labels(self):
        data = []
        tokens = list(nltk.word_tokenize(self.text))
        token_spans = textspan.get_original_spans(tokens, self.text)
        for token, span in zip(tokens, token_spans):
            if len(span) > 0:
                sbegin, send = span[0]
                label = 'O'
                for srl in self.srls + self.emotions:
                    if sbegin >= srl.cbegin and send <= srl.cend:
                        if srl.type == 'emotion':
                            label = 'emotion'
                        else:
                            label = srl.label
                data.append((token, label))
        return data
    
    def get_emotion_labels(self):
        data = []
        tokens = list(nltk.word_tokenize(self.text))
        token_spans = textspan.get_original_spans(tokens, self.text)
        for token, span in zip(tokens, token_spans):
            if len(span) > 0:
                ebegin, eend = span[0]
                label = 'O'
                for em in self.emotions:
                    if ebegin >= em.cbegin and eend <= em.cend:
                        label = em.label
                data.append((token, label))
        return data


class Dataset:
    def __init__(self, datadir, split):
        self.datadir = datadir
        self.docs = []
        self.split = split # train, dev, test, or None 

    def __repr__(self):
        return '\n'.join((str(d) for d in self.docs))
    
    def __iter__(self):
        self.n = -1
        return self

    def __next__(self):
        if self.n < len(self.docs):
            self.n += 1
            return self.docs[self.n-1]
        else:
            self.n = -1
            raise StopIteration
    
    def __getitem__(self, i):
        return self.docs[i]
    
    def __len__(self):
        return len(self.docs)


class RemanData(Dataset):
    def __init__(self, datadir, split):
        super().__init__(datadir, split)

    def normalize_emotion(self, unnormalized_emotion):
        if unnormalized_emotion in REMAN_EMOTION_MAPPING:
            return REMAN_EMOTION_MAPPING[unnormalized_emotion]
        return None

    def normalize_srl(self, unnormalized_srl):
        if unnormalized_srl in REMAN_SRL_MAPPING:
            return REMAN_SRL_MAPPING[unnormalized_srl]
        return None

    def read_data(self):
        tree = ET.parse(os.path.join(self.datadir, 'reman', 'reman-version1.0.xml'))
        root = tree.getroot()
        if self.split is not None:
            doc_ids = open(os.path.join(self.datadir, 'reman', f'{self.split}_ids.txt')).readlines()
            doc_ids = [d.strip() for d in doc_ids]
        else:
            doc_ids = None

        keep_doc = lambda did : True if self.split is None else (did in doc_ids)

        for doc in root:
            if keep_doc(doc.attrib['doc_id']):
                text = doc[0].text

                spans = doc[1][0]
                formatted_spans = []
                for s in spans:
                    emotion = self.normalize_emotion(s.attrib['type'])
                    if emotion is not None:
                        span = Span(s.attrib['cbegin'], s.attrib['cend'], emotion, "emotion")
                        formatted_spans.append(span)

                relations = doc[1][1]
                formatted_relations = []
                for s in relations:
                    srl = self.normalize_srl(s.attrib['type'])
                    if srl is not None:
                        span = Span(s.attrib['right'], s.attrib['left'], s.attrib['type'], "srl")
                        formatted_relations.append(span)

                self.docs.append(Document(doc.attrib, text, formatted_spans, formatted_relations))

class TweetData(Dataset):
    def __init__(self, datadir, split):
        super().__init__(datadir, split)

    def normalize_emotion(self, unnormalized_emotion):
        if unnormalized_emotion in TWEET_EMOTION_MAPPING:
            return TWEET_EMOTION_MAPPING[unnormalized_emotion]
        return None

    # Edit this function to change which metadata we keep
    def format_metadata(self, doc):
        return {
            "unitid": doc["unitid"],
            "createdat": doc["createdat"],
            "id": doc["id"],
            "country": doc["country"]
            }

    def read_data(self):
        df1 = pd.read_csv(os.path.join(self.datadir, 'ElectoralTweetsData/Annotated-US2012-Election-Tweets/Questionnaire2/Batch1/AnnotatedTweets.txt'),
                        sep='\t', error_bad_lines=False
                        )
        df2 = pd.read_csv(os.path.join(self.datadir, 'ElectoralTweetsData/Annotated-US2012-Election-Tweets/Questionnaire2/Batch2/AnnotatedTweets.txt'),
                        sep='\t', error_bad_lines=False
                        )
        df = pd.concat([df1,df2], axis=0, join="outer", ignore_index=True)

        # Column name aliases
        q1 = 'q1whoisfeelingorwhofeltanemotioninotherwordswhoisthesourceoftheemotion'
        q2 = 'q2whatemotionchooseoneoftheoptionsfrombelowthatbestrepresentstheemotion'
        q6 = 'q6towardswhomorwhatinotherwordswhoorwhatisthetargetoftheemotion'
        q7 = 'fontcolorolivetweetertweetfontbrq7whichwordsinthetweethelpidentifyingtheemotion'

        # Normalize emotions
        df[q2] = df[q2].apply(self.normalize_emotion)

        df.dropna(subset=['tweet', q1, q2, q6, q7], inplace=True)


        if self.split is not None:
            doc_ids = open(os.path.join(self.datadir, "ElectoralTweetsData", 
                            "Annotated-US2012-Election-Tweets", "Questionnaire2", 
                            f'{self.split}_ids.txt')).readlines()
            doc_ids = [d.strip() for d in doc_ids]
        else:
            doc_ids = None

        keep_doc = lambda did : True if self.split is None else (did in doc_ids)
        for _, doc in df.iterrows():
            if keep_doc(doc['unitid']):
                text = "_tweeter_ " + doc['tweet']

                em_start = text.find(doc[q7])
                em_end = em_start + len(doc[q7])
                emotion = [Span(em_start, em_end, doc[q2], "emotion")] 

                ex_start = 0 if doc[q1].lower() == 'tweeter' else text.find(doc[q1])
                ex_end = 9 if ex_start == 0 else (ex_start + len(doc[q1]))
                experiencer = Span(ex_start, ex_end, 'experiencer', "srl")

                tg_start = 0 if doc[q6].lower() == 'tweeter' else text.find(doc[q6])
                tg_end = 9 if tg_start == 0 else (tg_start + len(doc[q6]))
                target = Span(tg_start, tg_end, 'target', "srl")
                srls = [experiencer, target]

                self.docs.append(Document(self.format_metadata(doc), text, emotion, srls))

if __name__ == '__main__':
    reman_data = RemanData('data', None)
    reman_data.read_data()
    tweet_data = TweetData('data', None)
    tweet_data.read_data()