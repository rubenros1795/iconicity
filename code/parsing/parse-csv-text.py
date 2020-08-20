from boilerpipe.extract import Extractor
from http.client import IncompleteRead
from htmldate import find_date
from bs4 import BeautifulSoup
import concurrent.futures
import concurrent
import threading
from lxml import html
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import urllib.request
import pandas as pd
import re as regexz
import codecs
import datetime
import requests
import os.path
import string
import random
import shutil
import json
import math
import time
import nltk
import glob
import uuid
import csv
import io

class textparser():
    def __init__(self,html_folder):
        self.html_folder = html_folder

    def Import(html_folder):

        if os.path.exists(os.path.join(html_folder, "results.txt")):
            with open(os.path.join(html_folder, "results.txt")) as f:
                urls = f.readlines()
                urls_new = []
                for u in urls:
                    try:
                        url = str(u).split('|')[1].replace('\n','')
                        id_ = str(u).split('|')[0]
                        id_ = os.path.split(id_)[-1].split('html')[-1][0:]
                        id_ = os.path.join(html_folder,id_ + ".html")
                        urls_new.append((id_, url))
                    except IndexError:
                        continue
                return urls_new
        else:
            print("results.txt not found in {}".format(html_folder))

    def ParserBoilerArticle(html_object):
        extractor = Extractor(extractor='ArticleSentencesExtractor', html=html_object)
        sents = extractor.getText()
        try:
            return sents
        except Exception as e:
            return

    def ParserBoilerDefault(html_object):
        extractor = Extractor(extractor='DefaultExtractor', html=html_object)
        sents = extractor.getText()
        try:
            return sents
        except Exception as e:
            return

    def ParserBoilerEverything(html_object):
        extractor = Extractor(extractor='DefaultExtractor', html=html_object)
        sents = extractor.getText()
        try:
            return sents
        except Exception as e:
            return

    def ParserRaw(html_object):
        soup = BeautifulSoup(html_object, "html.parser")
        [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
        text = soup.getText()
        #text = [t for t in text if t]
        return text

    def Parse(html_folder):
        print(html_folder)
        urls = textparser.Import(html_folder)
        print("INFO: parsing text from {} files".format(len(urls)))
        destination_path = os.path.join(html_folder[:-4], "txt")
        
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        tsv_fn = os.path.join(destination_path, "parsed_text.tsv")
        pd.DataFrame(columns = "id text".split(' ')).to_csv(tsv_fn,sep="\t",index=False)

        
        with open(tsv_fn, "a",encoding='utf-8') as fp:
            for u in tqdm(urls):
                fn = os.path.join(html_folder, u[0])
                url = u[1]
                id_ = "_".join([fn,url])
                try:
                    with codecs.open(fn,'r',encoding='utf-8',errors='ignore') as f:
                        html_object = f.read()

                    sents = textparser.ParserBoilerArticle(html_object)

                    if len(sents) < 2 or sents is None:
                        sents = textparser.ParserBoilerDefault(html_object)
                    if len(sents) < 2 or sents is None:
                        sents = textparser.ParserBoilerEverything(html_object)
                    if len(sents) < 2 or sents is None:
                        sents = textparser.ParserRaw(html_object)

                    if type(sents) == list:
                        row = "\t".join([id_," ".join([str(sent) for sent in sents]).replace('\t',' ').replace('\n',' ')])
                        fp.write(row + '\n')
                    if type(sents) == str:
                        row = "\t".join([id_,sents.replace('\t',' ').replace('\n',' ')])
                        fp.write(row + '\n')

                except Exception as e:
                    print(e)
                    continue
        

######################

def Prepare(photo,datapath):

    photo_directory = os.path.join(datapath,photo)
    html_subdirectory = os.path.join(datapath,photo, "html")
    txt_subdirectory = os.path.join(datapath,photo, "txt")
    metadata_subdirectory = os.path.join(datapath,photo, "metadata")

    for dir_ in [photo_directory, html_subdirectory, txt_subdirectory, metadata_subdirectory]:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

def LoadData(tsv_path,mode):
    df = pd.read_csv(tsv_path,sep='\t')

    if mode == "list":
        df = df[["filename","page_url"]]
        df["filename"] = [os.path.split(x)[-1] for x in list(df['filename'])]
        return list(df['page_url'])
    if mode == "tsv":
        return df

#######################
def Pipe(photo,tsv_path,datapath):

    #Parse Texts
    textparser.Parse(os.path.join(datapath,photo,"html"))

#####################


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()

    # parser.add_argument('-f', '--photo', dest="photo", required=True)

    # args = parser.parse_args()

    # photos = args.photo
    # photos = photos.split("_")
    
    #for photo in "Monk NapalmGirl Plane911 Rwanda|| SharbatGula TankMan VietCong WarRoom".split(' '):
    for photo in ['TimesSquareKiss']:
        print(f'working on {photo}')
        Pipe(
            photo=photo,
            tsv_path= os.path.join("F:/","react-data","iconic","tsv",photo + ".tsv"),
            datapath=os.path.join("F:/","react-data","iconic","tsv")
            )
