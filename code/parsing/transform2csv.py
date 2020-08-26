import os
import json
import pandas as pd 
import csv
import time
from tqdm import tqdm
import langid
from langid.langid import LanguageIdentifier, model
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

base_path = "path/to/photo/folder"

## Functions for Whole File
def GetLabels(json_file):
    lbs = []
    try:
        if 'responses' in json_file.keys() and 'webDetection' in json_file['responses'][0].keys() and 'bestGuessLabels' in json_file['responses'][0]['webDetection'].keys():
            for l in json_file['responses'][0]['webDetection']['bestGuessLabels']:
                lbs.append(l['label'])
            return "||".join(lbs)
        else:
            return "na"
    except Exception as e:
        return "na"

def GetEntities(json_file):
    ents = []
    try:
        if 'responses' in json_file.keys() and 'webDetection' in json_file['responses'][0].keys() and 'webEntities' in json_file['responses'][0]['webDetection'].keys():
            for l in json_file['responses'][0]['webDetection']['webEntities']:
                
                if 'description' in l.keys() and 'entityId' in l.keys() and 'score' in l.keys():
                    description = l['description']
                    entId = l['entityId']
                    score = l['score']

                    e = f'{description}_({entId})_({score})'
                    ents.append(e)
            return "||".join(ents)
        else:
            return "na"
    except Exception as e:
        return "na"

## Functions for Every PwMI instance
def GetPageURL(PwMi):
    try:
        if 'url' in PwMi.keys():
            PageURL = PwMi['url']
            return PageURL
        else:
            return "na"
    except Exception as e:
        return "na"

def GetPageTitle(PwMi):
    try:
        if 'pageTitle' in PwMi.keys():
            pageTitle = PwMi['pageTitle']
            return pageTitle.replace('\t','')
        else:
            return "na"
    except Exception as e:
        return "na"

def GetImgFull(PwMi):
    urls = []
    try:
        if 'fullMatchingImages' in PwMi.keys():
            for fi in PwMi['fullMatchingImages']:
                urls.append(fi['url'])
            return "||".join(urls)
        else:
            return "na"
    except Exception as e:
        return "na"

def GetImgPartial(PwMi):
    urls = []
    try:
        if 'partialMatchingImages' in PwMi.keys():
            for fi in PwMi['partialMatchingImages']:
                urls.append(fi['url'])
            return "||".join(urls)
        else:
            return "na"
    except Exception as e:
        return "na"

## Master
def Transform2Csv(photo):
    #########################
    photo_folder = os.path.join(base_path, photo)
    num_iterations = len([fol for fol in os.listdir(photo_folder) if os.path.isdir(os.path.join(photo_folder,fol)) and "source" not in fol and "context" not in fol])
    range_iter = [str(i) for i in list(range(1,num_iterations+1))]
    print(photo,range_iter)
    folder_base = os.path.join(base_path,photo,photo)
    #########################

    # Create TSV
    
    tsv_fn = os.path.join(base_path, "tsv", photo + ".tsv")
    pd.DataFrame(columns = "filename photo page_url image_url_full image_url_partial page_title iteration language labels scrape_date".split(' ')).to_csv(tsv_fn,sep="\t",index=False)

    # Loop
    processed_urls = set()


    with open(tsv_fn, 'a',encoding='utf-8') as f:
        for iteration in range_iter:
            print(photo,iteration)
            list_json = [os.path.join(folder_base + "_" + iteration,x) for x in os.listdir(folder_base + "_" + iteration) if ".json" in x]
            
            for jsfn in tqdm(list_json):
                try:
                    with open(jsfn) as fp:
                        json_file = json.load(fp)
                except Exception as e:
                    print(e)
                    continue
                labels = GetLabels(json_file)
                if labels == None:
                    labels = "na"

                if 'responses' in json_file.keys() and 'webDetection' in json_file['responses'][0].keys() and 'pagesWithMatchingImages' in json_file['responses'][0]['webDetection'].keys():
                    for pwmi in json_file['responses'][0]['webDetection']['pagesWithMatchingImages']:
                        PageURL=GetPageURL(pwmi)
                        if PageURL not in processed_urls:

                            PageTitle=GetPageTitle(pwmi)
                            if PageTitle == None:
                                PageTitle = "na"

                            ImgFull=GetImgFull(pwmi)
                            if ImgFull == None:
                                ImgFull = "na"

                            ImgPartial=GetImgPartial(pwmi)
                            if ImgPartial == None:
                                ImgPartial = "na"

                            scrape_date = os.path.getmtime(jsfn)
                            scrape_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(scrape_date))

                            language =  "=".join([str(identifier.classify(PageTitle)[0]),str(identifier.classify(PageTitle)[1])])

                            fields = "\t".join([jsfn,photo,PageURL,ImgFull,ImgPartial,PageTitle,str(iteration),language,labels,str(scrape_date)])
                            
                            if fields == "\t".join("na na na na na na na na na na".split(' ')):  
                                continue
                            f.write(fields + '\n')
                            processed_urls.add(PageURL)
                        else:
                            continue


list_photos = ['']
print(len(list_photos))

for photo in list_photos:
    Transform2Csv(photo)

