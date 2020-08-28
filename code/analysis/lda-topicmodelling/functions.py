import os
import pandas as pd
from tqdm import tqdm
import re
import langid 
import random
import numpy as np
import string
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Phrases

import spacy

class Data():

    def __init__(self):
        self.str_ = str_
        self.filename = filename
        self.photo = photo
        self.sample_size = sample_size

    def readCsv(filename):
        with open(filename,'r',encoding='utf-8') as f:
            df = f.readlines()
            cols = df[0].replace('\n','').split('\t')
            df = [x.replace('\n','').split('\t') for x in df[1:]]
            df = pd.DataFrame(df).iloc[:,:12]
            df.columns = cols
        return df

    def readText(filename):
        with open(filename,'r',encoding='utf-8') as f:
            df = f.readlines()
            cols = df[0].replace('\n','').split('\t')
            df = [x.replace('\n','').split('\t') for x in df[1:]]
        return df

    def readInd(filename):
        with open(filename,'r',encoding='utf-8') as f:
            return f.read()

    def readTextFiles(photo,sample_size):
        files = f"path/tp/tsv/{photo}/txt/files"
        files = [os.path.join(files,x) for x in os.listdir(files)]
        if len(files) > sample_size:
            files = random.sample(files,sample_size)
        files = {f:Data.readInd(f) for f in tqdm(files)}
        return files


    def cleanHtml(str_):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', str(str_))
        s = re.sub(r'[^A-Za-z0-9 ]+', ' ', cleantext)
        s = re.sub(' +', ' ', s)
        s = s.lower().split(' ')
        return [w for w in s if w]
    
    def cleanText(str_):
        #strip_special_chars = re.compile("[^A-Za-z0-9#]+")
        translator = str.maketrans('', '', string.punctuation)
        txt = str_.translate(translator)
        txt = re.sub('\s+', ' ', txt).strip()
        txt = txt.lower()
        return txt.split(' ')

class PreProcess():

    def __init__(self):
        self.texts = texts

    def sent_to_words(texts):
        for sentence in texts:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    def remove_stopwords(texts):
        stop_words = stopwords.words('english')
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def TopicTerms(lda_model,num_topics,topic):
        tm = dict(lda_model.show_topics(formatted=False,num_words=15,num_topics=num_topics))
        ttt = [i[0] for i in tm[int(topic)]]
        return " | ".join(ttt)

    def make_bigrams(texts):
        bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.
        bigram_mod = gensim.models.phrases.Phraser(bigram)

        return [bigram_mod[doc] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        nlp = spacy.load('en', disable=['parser', 'ner'])
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

class TM():

    def __init__(self):
        self.texts = texts   
    
    def DictionaryCorpus(texts):
        # Create Dictionary
        id2word = corpora.Dictionary(texts)
        corpus = [id2word.doc2bow(text) for text in texts]
        return id2word,corpus