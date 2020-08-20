from boilerpipe.extract import Extractor
from concurrent.futures import ThreadPoolExecutor

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

###### helper classes
class pagescraper():
    def __init__(self,destination_path, list_url,filename):
        self.destination_path = destination_path
        self.list_url = list_url
        self.filename = filename

    # def Scraper(url, destination_path):
    #     title = os.path.join(destination_path, str(uuid.uuid4()) + '.html')
    #     resp = requests.get(url, verify=False, timeout=30)

    #     with open(title, "wb") as fh:
    #         fh.write(resp.content)

    #     with open(os.path.join(destination_path,"results.txt"), "a") as fh:
    #         fh.write("{}|{}{}".format(title[:-5], url, '\n'))

    #     time.sleep(0.01)

    def Scraper(tpl):
        url = tpl[0]
        destination_path = tpl[1]
        resp = requests.get(url, verify=False, timeout=30)

        with open(destination_path, "wb") as fh:
            fh.write(resp.content)
        time.sleep(0.01)


    # def PoolScrape(list_url, destination_path):

    #     with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    #         # Start the load operations and mark each future with its URL
    #         future_to_url = {executor.submit(pagescraper.Scraper, url, destination_path): url for url in list_url}
    #         for future in concurrent.futures.as_completed(future_to_url):
    #             url = future_to_url[future]
    #             try:
    #                 data = future.result()
    #             except Exception as exc:
    #                 #print(exc)
    #                 pass
    #             else:
    #                 pass

    def ThreadScraper(list_tpls):
        with ThreadPoolExecutor() as e:
            for u in list_tpls:
                e.submit(pagescraper.Scraper, u)


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
    # with open(tsv_path,'r',encoding='utf-8') as f:
    #     df = f.readlines()
    #     cols = df[0].replace('\n','').split('\t')
    #     df = [x.replace('\n','').split('\t') for x in df[1:]]
    #     df = pd.DataFrame(df).iloc[:,:12]
    #     df.columns = cols
    df = pd.read_csv(tsv_path,sep='\t')

    if mode == "tuple":
        return list(df['page_url'])
    if mode == "tsv":
        return df

#######################
def Pipe(photo,tsv_path,datapath):

    Prepare(photo,datapath)
    data = LoadData(tsv_path,mode="tuple")
    
    destination_path = os.path.join(datapath,photo,"html")
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    data = [[u,os.path.join(destination_path,str(uuid.uuid4()) + '.html')] for u in data]
    print(f"{len(data)} urls found in photo {photo}")
    
    with open(os.path.join(destination_path,"results.txt"),'w',encoding='utf-8') as f:
        for tpl in data:
            f.write(tpl[1] + "|" + tpl[0] + "\n")

    pagescraper.ThreadScraper(data)
    

#####################


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()

    # parser.add_argument('-f', '--photos', dest="photos", required=True)

    # args = parser.parse_args()

    # photo = args.photos

        
    Pipe(
        photo='iraq',
        tsv_path=f"F:/react-data/protest/iraq/iraq.tsv",
        datapath="F:/react-data/protest/iraq/"
        )
