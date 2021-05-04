import tqdm
from nltk.corpus import stopwords

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