# Code and Data for 'Quantifying Iconicity in 940K Online Circulations of 26 Iconic Photographs'

This repository contains the code and data for the research presented in "Quantifying iconicity in 1.1M online circulations of 26 iconic photographs" (Smits & Ros, 2020). Find the paper [here](http://ceur-ws.org/Vol-2723/short34.pdf).

Included in this repository are scripts for parsing data outputted by the Google Cloud Vision API and scripts used for the analysis of textual and visual material.

In the paper we use clustered document embeddings to map the different type of textual context around online reproductions of iconic images. The method is described in the paper. This repo also contains the code for:

#### Webpage text classification using latent Dirichlet allocation (LDA) topic modelling
We train separate topic models for every set of URLs associated with a specific iconic image. Although we are aware of the work in multilingual topic modelling, a large share of our corpus (around 70\%) consists of English-language URLs, and we therefore choose to report in this article only the results for this subset. Preprocessing entails tokenization, lemmatization and the removal of stopwords. We use a combination of qualitative evaluation and coherence measures,based on normalized pointwise mutual information and cosine similarity, to determine a number of topics for every subset.

#### Optical Character Recognition (OCR)
Iconic images are frequently supplemented with text for various reasons. To map this aspect of the online reproduction, we run Optical Character Recognition on all the images, using the powerful Tesseract OCR engine.
  
#### Identification of frequently used 'areas' of the original image using SIFT
Online reproductions of iconic images often use specific parts of the original image. To detect crops and regions of interest, we use Scale-Invariant Feature Transform (SIFT) to identify the keypoints of the original image and inspect which part of the original image is used in a reproduction. We first preprocessed our corpus by converting the image to grayscale, downsampling the images and applying a Gaussian blur.
  
#### Identification of images frequently combined with the iconic image
The context of online republications of iconic images is not only textual. We aggregate the visual context of online reproduction by identifying and clustering all other images published on a webpage. We also look at images that combine the iconic image with another image (in a single image file).
