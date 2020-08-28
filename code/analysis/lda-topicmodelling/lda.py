import re
import numpy as np
import pandas as pd

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# NLTK Stop words
from nltk.corpus import stopwords
from collections import Counter
import random
import json
import os
import langid

from functions import *
import pickle


def Prepare(photo,sample_n):
    data = Data.readTextFiles(photo,sample_n)
    data = {os.path.split(k)[-1].replace('.txt',''):" ".join(Data.cleanText(v)) for k,v in tqdm(data.items()) if langid.classify(v)[0] == "en"}
    texts = list(data.values())
    texts = [t.split(' ') for t in texts]
    texts = [t for t in texts if len(t) > 100 and len(t) < 100000]
    print('\t -- finished loading texts')
    texts = PreProcess.make_bigrams(texts)
    texts = PreProcess.remove_stopwords(texts)
    texts = PreProcess.lemmatization(texts)
    print('\t -- finished preprocessing')
    id2word,corpus = TM.DictionaryCorpus(texts)
    print('\t -- created dictionary and corpus in Gensim format')
    return texts,id2word,corpus

def LDA(photo,n_topics,id2word,corpus,texts):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,
                                               num_topics=n_topics,
                                               random_state=100,update_every=1,chunksize=100,
                                               passes=100,alpha='auto',
                                               per_word_topics=True)
    print('\t -- finished training')
    fn_model = os.path.join('path/to/output/topicmodels',f"lda-{photo}-{n_topics}-model.pkl")
    fn_dictionary = os.path.join('path/to/output/topicmodels',f"lda-{photo}-{n_topics}-dictionary.pkl")
    fn_corpus = os.path.join('path/to/output/topicmodels',f"lda-{photo}-{n_topics}-corpus.json")

    # Save top words
    top_words_per_topic = []
    for t in range(lda_model.num_topics):
        top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn = 15)])

    pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv(f"path/to/output/topicmodels/top_words-{photo}-{n_topics}.csv",index=False)

    # Save model
    with open(fn_model,'wb') as f:
        pickle.dump(lda_model, f)

    # Save dictionary
    id2word.save(fn_dictionary)

    # Save corpus
    with open(fn_corpus,'w',encoding='utf-8') as f:
        json.dump(corpus,f)
    print('\t -- saved everything')

#########
list_photos = "".split(' ')
for photo in list_photos:
    try:
        print("Working on",photo)
        sample_n = 10000
        texts,id2word,corpus = Prepare(photo,sample_n)
        fn_texts = os.path.join('path/to/texts',f"lda-{photo}-texts.json")
        # Save texts
        with open(fn_texts,'w',encoding='utf-8') as f:
            json.dump(texts,f)
        for n_topics in [4,6,8,12,16]:
            print(f"\t working on n_topics {n_topics}")
            LDA(photo,n_topics,id2word,corpus,texts)
    except Exception as e:
        print(e)
        continue