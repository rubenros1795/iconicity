import cv2 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from skimage import color
import pandas as pd
import glob, os
from numpy.linalg import norm
from tqdm import tqdm
import random
import requests, io
import glob
import concurrent.futures
import shutil
from PIL import Image

class preprocess():
    
    def __init__(img):
        self.img = img

    def TransformGray(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_gray = color.rgb2gray(img)
        return img_gray

    def TransformBlur(img):
        img_blur = cv2.GaussianBlur(img,(15,15),0)
        return img_blur

    def TransformResize(img):
        if img.shape[1] > 500:
            ratio_ = 500 / img.shape[1]
            height = img.shape[0] * ratio_
            dim = (500, int(height))
            # resize image
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            return resized
        else:
            return img
        
    def Transform(img):
        img = preprocess.TransformResize(img)
        img = preprocess.TransformGray(img)
        img = preprocess.TransformBlur(img)
        return img

def parse(tpl):
    filename = tpl[1]
    photo = tpl[0]
    img = cv2.imread(filename)
    clean_img = preprocess.Transform(img)
    new_fn = os.path.split(filename)[-1]
    new_fn = os.path.join(f"path/to/preprocessed/images/photo",new_fn)
    
    cv2.imwrite(new_fn,clean_img)

def Scrape(list_tpls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
        for u in list_tpls:
            e.submit(parse, u)
### 

if __name__ == "__main__":
    list_photos = [i[:-4] for i in os.listdir('path/to/tsv/') if '.tsv' in i]
    for photo in list_photos:
        df = pd.read_csv(f"path/to/sift/test/results.csv",sep=";")
        df = list(df[df['correct'] == "True"]["filename"])
        directory_cleaned = f"F:/react-data/iconic/image-analysis/test_results/{photo}/preproc"
        if not os.path.exists(directory_cleaned):
            os.makedirs(directory_cleaned)

        list_photos = [(photo,fn) for fn in df]
        Scrape(list_photos)
