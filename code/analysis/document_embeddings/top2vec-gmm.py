from functions import *
import os,json
import pandas as pd
import argparse


basepath = "F:/react-data/iconic/tsv/"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('photo', type=str, help='Input photo')
    args = parser.parse_args()
    photo = args.photo
    sample_size = 100000

    print(f"working on photo {photo} with a sample size of {sample_size}")

    texts = Data.Import(photo,sample_size)
    print(f"{len(texts)} left for training doc2vec")
    with open(f"models/data-{photo}-doc2vec.json", 'w') as f:
        json.dump(texts, f)

    model = Embeddings.TrainModel(texts)
    model.save(f"models/doc2vec-{photo}-e75.model")

    # umap_model = Cluster.DrUmap(model)
    # gmm_results = Cluster.GMM(model,umap_model)
    # gmm_results.to_csv('doc2vec-gmm-results.csv',index=False)