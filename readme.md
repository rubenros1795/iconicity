# Code and Data for 'Quantifying Iconicity'

This repository contains the code and data for the research presented in "Quantifying iconicity in 1.1M online circulations of 26 iconic photographs" (Smits & Ros, 2020). 

Included in this repository are scripts for parsing data outputted by the Google Cloud Vision API and scripts used for the analysis of textual and visual material.

In the paper, we focus on several types of analysis:

- document embedding models trained on the text parsed from the webpages identified as including an iconic image.

  `We use doc2vec to generate embedding models for every set of webpages associated with an iconic image, based on the top2vec method. We subsequently cluster the embeddings using Gaussian Mixture Models (GMM) clustering, which returns a probability distribution for every document. Before clustering, we reduce the high dimension embeddings with UMAP. The number of clusters is determined by HDBSCAN clustering. Because doc2vec also trains word embeddings, we calculate an average (document) vector for every topic and identify the words closest to this vector`

- LDA topic modelling.

- analysis of images:

  - Optical Character Recognition
  - identification of frequently used 'areas' of the original image using SIFT 
  - identification of images combined with the original image
  - context image analysis