from PIL import Image
import pandas as pd
import pytesseract
import random
import argparse
import concurrent.futures
import cv2
import os

def ocr(tpl):
    page_url = tpl[0]
    image_url = tpl[1]
    photo = tpl[2]
    filename = tpl[3]

    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    new_filename = os.path.split(filename)[-1]
    id_ = new_filename.split('.')[0]
    cv2.imwrite(new_filename, gray)
    text = pytesseract.image_to_string(Image.open(new_filename))
    
    os.remove(new_filename)

    if text:
        if len(text) > 0:
            with open(f"F:/react-data/iconic/image-analysis/ocr/{photo}/{id_}.txt", "w",encoding='utf-8') as fp:
                fp.write(text)

def Scrape(list_tpls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
        for tpl in list_tpls:
            e.submit(ocr, tpl)

if __name__ == "__main__":
    
    list_photos = []
    for photo in list_photos:
        # Subset
        df = pd.read_csv(f'F:/react-data/iconic/image-analysis/test_results/{photo}.csv',sep=';')
        df = df[df['correct'] == "True"].reset_index()
        #fn_results = f"F:/react-data/iconic/image-analysis/ocr/{photo}-ocr.tsv"
        #pd.DataFrame(columns = "page_url image_url photo filename ocr".split(' ')).to_csv(fn_results,sep="\t",index=False)
        
        directory_cleaned = f"F:/react-data/iconic/image-analysis/ocr/{photo}"
        if not os.path.exists(directory_cleaned):
            os.makedirs(directory_cleaned)
        
        list_tpls = []
        for c,i in enumerate(df['index']):
            page_url = df['page_url'][c]
            image_url = df['image_url'][c]
            photo = df['photo'][c]
            page_url = df['page_url'][c]
            filename = df['filename'][c]
            list_tpls.append((page_url,image_url,photo,filename))
        Scrape(list_tpls)