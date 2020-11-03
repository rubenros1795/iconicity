import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
# import umap
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.cluster import dbscan
from sklearn.mixture import GaussianMixture
import os
import pandas as pd
import random
from tqdm import tqdm
import re
import csv
import string
import requests
import langid
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
from gensim.test.utils import get_tmpfile

basepath = "D:/react-data/iconic/tsv/"


class Data():

    def __init__(self):
        self.filename = filename
        self.photo = photo
        self.sample_size = sample_size
        self.basepath = basepath


    def rt(filename):
        with open(filename,'r',encoding='utf-8') as f:
            return f.read()
            
    def ReadTexts(photo,sample_size):
        # fulldf = 
        files = f"{basepath}/{photo}/txt/files_corrected"
        files = [os.path.join(files,x) for x in os.listdir(files)]
        if len(files) > sample_size:
            files = random.sample(files,sample_size)
        files = {f:Data.rt(f) for f in tqdm(files)}
        return files

    def Tokenize(doc):
        """Tokenize documents for training and remove too long/short words"""
        return simple_preprocess(strip_tags(doc), deacc=True)

    def Import(photo,sample_size):
        print(' tokenizing....')
        t = Data.ReadTexts(photo,sample_size)
        print(' reading languages + subsetting....')
        t = [Data.Tokenize(v) for k,v in tqdm(t.items()) if langid.classify(v)[0] == 'en']
        t = [x for x in t if len(x) > 50]
        t = [TaggedDocument(doc, [i]) for i, doc in enumerate(t)]
        print(f"after tokenizing and subsetting: {len(t)} documents left")
        return t

class Embeddings():
    def __init__(self):
        self.texts = texts
        self.basepath = "F:/react-data/iconic/tsv/"


    def TrainModel(texts):
        print(' training doc2vec model....')
        model = Doc2Vec(documents=texts,
                 vector_size=300,
                 min_count=75, window=12,
                 sample=1e-5,
                 negative=5,
                 hs=1,
                 epochs=75,
                 dm=0,
                 dbow_words=1,
                 workers=4)
        return model


class Cluster():
    def __init__(self):
        self.model = model

    # def DrUmap(model):
    #     umap_model = umap.UMAP(n_neighbors=15,
    #                    min_dist=0.0,
    #                    n_components=5,
    #                    metric='cosine').fit(model.docvecs.vectors_docs)
    #     return umap_model

    def ClusNum(vectors):
        # word_cosine = cosine_distances(vectors)
        affprop = AffinityPropagation(max_iter=1000)
        af = affprop.fit(vectors)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_

        no_clusters = len(cluster_centers_indices)
        return no_clusters

    def _create_topic_vectors(model,cluster_gmm,num_clusters):
        topic_vectors = np.vstack([model.docvecs.vectors_docs[[c for c,i in enumerate(cluster_gmm) if np.argmax(i) == x]].mean(axis=0)
                                for x in range(num_clusters)])
        return topic_vectors

    def _find_topic_words_scores(model,topic_vectors):
        topic_words = []
        topic_word_scores = []

        for topic_vector in topic_vectors:
            sim_words = model.wv.most_similar(positive=[topic_vector], topn=50)
            topic_words.append([word[0] for word in sim_words])
            topic_word_scores.append([round(word[1], 4) for word in sim_words])

        topic_words = np.array(topic_words)
        topic_word_scores = np.array(topic_word_scores)
        return topic_words,topic_word_scores
    
    def GMM(model,umap_model):
        num_clusters = Cluster.ClusNum(umap_model.embedding_)
        print(f' found {num_clusters} with Affinity Propagation on umap embeddings')
        if num_clusters == 0:
            num_clusters = 3
        gmm = GaussianMixture(n_components=num_clusters)
        gmm.fit(umap_model.embedding_)
        cluster_gmm = gmm.predict_proba(umap_model.embedding_)
        topic_vectors = Cluster._create_topic_vectors(model,cluster_gmm,num_clusters)
        topic_words,topic_word_scores = Cluster._find_topic_words_scores(model,topic_vectors)
        data = [[c," ".join(t[:8]),sum([x[c] for x in cluster_gmm])] for c,t in enumerate(topic_words)]
        data = pd.DataFrame(data,columns=['cluster','words','prominence'])
        return data