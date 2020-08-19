# Code and Data for 'Quantifying Iconicity'

This repository contains the code and data for the research presented in "Quantifying iconicity in 1.1M online circulations of 26 iconic photographs" (Smits & Ros, 2020). 

Included in this repository are scripts for parsing data outputted by the Google Cloud Vision API and scripts used for the analysis of textual and visual material.

In the paper, we focus on several types of analysis:

- document embedding models trained on the text parsed from the webpages identified as including an iconic image.

  `We use doc2vec to generate embedding models for every set of webpages associated with an iconic image, based on the top2vec method. We subsequently cluster the embeddings using Gaussian Mixture Models (GMM) clustering, which returns a probability distribution for every document. Before clustering, we reduce the high dimension embeddings with UMAP. The number of clusters is determined by HDBSCAN clustering. Because doc2vec also trains word embeddings, we calculate an average (document) vector for every topic and identify the words closest to this vector`.

- LDA topic modelling.

  `We train separate topic models for every set of URLs associated with a specific iconic image. Although we are aware of the work in multilingual topic modelling, a large share of our corpus (around 70\%) consists of English-language URLs, and we therefore choose to report in this article only the results for this subset. Preprocessing entails tokenization, lemmatization and the removal of stopwords. We use a combination of qualitative evaluation and coherence measures,based on normalized pointwise mutual information and cosine similarity, to determine a number of topics for every subset. `

- analysis of images:

  - Optical Character Recognition
  
    `Iconic images are frequently supplemented with text for various reasons. To map this aspect of the online reproduction, we run Optical Character Recognition on all the images, using the powerful Tesseract OCR engine.` 
  
  - identification of frequently used 'areas' of the original image using SIFT 
  
    `Online reproductions of iconic images often use specific parts of the original image. To detect crops and regions of interest, we use Scale-Invariant Feature Transform (SIFT) to identify the keypoints of the original image and inspect which part of the original image is used in a reproduction. We first preprocessed our corpus by converting the image to grayscale, downsampling the images and applying a Gaussian blur.`
  
  - identification of images combined with the original image
  
  - context image analysis