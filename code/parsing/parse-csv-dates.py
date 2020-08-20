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


class dateparser():
    def __init__(self,htmlpath,gatherFile):
        self.htmlpath = htmlpath
        self.gatherFile = gatherFile

    def Parse(htmlpath,gatherFile):
        try:
            with open(htmlpath, "r",encoding='utf-8') as f:
                page = f.read()
            tree = html.fromstring(page)
            date = find_date(tree,original_date=True)
            if date:
                with open(gatherFile,'a') as f:          
                    f.write(htmlpath+"||"+date+"\n")
        except Exception as e:
            return
        

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

    # Scrape Dates
    filename_dates = os.path.join(datapath,photo,"metadata","dates.txt")
    open(filename_dates,'w').close()

    for p in tqdm([os.path.join(os.path.join(datapath,photo,"html"),x) for x in os.listdir(os.path.join(datapath,photo,"html")) if ".html" in x]):
        dateparser.Parse(p,filename_dates)
    

#####################


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()

    # parser.add_argument('-f', '--photo', dest="photo", required=True)

    # args = parser.parse_args()

    # photo = args.photo
    Pipe(
        photo='2003',
        tsv_path='C:/Users/Ruben/Documents/GitHub/ReACT_GCV/data/images_tables_article_2003/data-full.tsv',
        datapath="F:/react-data/iconic/tsv/"
        )
