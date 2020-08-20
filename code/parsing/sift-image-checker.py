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


def LoadData(tsv_path):
    with open(tsv_path,'r',encoding='utf-8') as f:
        df = f.readlines()
        cols = df[0].replace('\n','').split('\t')
        df = [x.replace('\n','').split('\t') for x in df[1:]]
        df = pd.DataFrame(df).iloc[:,:12]
        df.columns = cols
    return df

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

class sift():
    def __init__(img1,img2):
        self.img1 = img1 
        self.img2 = img2 
        self.img_url = img_url
    
    def tester(img1,img2):
        try:
            sift = cv2.xfeatures2d.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1,None)
            kp2, des2 = sift.detectAndCompute(img2,None)
            index_params = dict(algorithm = 0, trees = 5)
            search_params = dict(checks = 50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1,des2,k=2)

            good = []
            for m,n in matches:
                if m.distance < 0.8*n.distance:
                    good.append(m)
                    
            if len(good)>10:
                return len(good),True
            else:
                return len(good),False
        
        except Exception as e:
            return "na","na"

    def process_image(tpl):
        
        img1 = tpl[0]
        fn = str(tpl[1])
        page_url = tpl[2]
        url = tpl[3]
        photo = tpl[4]

        with open(f"F:/react-data/iconic/image-analysis/test_results/{photo}.csv", "a",encoding='utf-8') as fp:
            if "png" in url:
                fn = str(fn) + ".png"
            elif "jpg" in url:
                fn = str(fn) + ".jpg"
            elif "Jpeg" in url:
                fn = str(fn) + ".jpg"
            elif "jpeg" in url:
                fn = str(fn) + ".jpg"
            elif "JPG" in url:
                fn = str(fn) + ".jpg"
            else:
                row = ";".join([page_url.replace(';',":"),url.replace(';',":"),photo,fn,"na","na"])
                fp.write(row + '\n')
                return "na","na"
            path_ = f"F:/react-data/iconic/image-analysis/test_results/{photo}"
            
            if not os.path.exists(path_):
                os.makedirs(path_)
            
            fn = os.path.join(path_,fn)
            image_content = requests.get(url, timeout=20).content
            image_file = io.BytesIO(image_content)
            image = Image.open(image_file).convert('RGB')
            with open(fn, 'wb') as f:
                image.save(f)
            img2 = cv2.imread(fn)
            img2 = preprocess.Transform(img2)
            len_matches,result = sift.tester(img1,img2)
            row = ";".join([page_url.replace(';',":"),url.replace(';',":"),photo,fn,str(len_matches),str(result)])
            fp.write(row + '\n')
            #os.remove(fn)
            return len_matches,result

def Scrape(list_tpls):
        with concurrent.futures.ThreadPoolExecutor() as e:
            for u in list_tpls:
                e.submit(sift.process_image, u)
### 

if __name__ == "__main__":
    #list_photos = [i[:-4] for i in os.listdir('F:/react-data/iconic/tsv/') if '.tsv' in i]
    for photo in ["TimesSquareKiss","VietCong","WarRoom"]:
        fn_results = f"F:/react-data/iconic/image-analysis/test_results/{photo}.csv"
        pd.DataFrame(columns = "page_url image_url photo filename number_matches correct".split(' ')).to_csv(fn_results,sep=";",index=False)
        df = LoadData(f"F:/react-data/iconic/tsv/{photo}.tsv")
        img1 = glob.glob(f"D:/react-data/iconic/{photo}/{photo}_source/*")[0]
        img1 = cv2.imread(img1)
        img1 = preprocess.Transform(img1)

        list_img_url = []
        for c,i in enumerate(df['page_url']):
                image_url = df['image_url_full'][c]
                if image_url == "na":
                    image_url = df['image_url_partial'][c]
                if "|" in image_url:
                    image_url = image_url.split('|')[0]
                list_img_url.append((img1,c,i,image_url,photo))

        #list_img_url = random.sample(list_img_url,10)
        Scrape(list_img_url)