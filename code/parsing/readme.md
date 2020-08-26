# Parsing

This subfolder contains several scripts used from parsing data from the Google Cloud Vision API output (.json files) to .tsv format. Start with ```transform2csv.py``` , which parses data into a .tsv with the following columns:

```filename photo page_url image_url_full image_url_partial page_title iteration language labels scrape_date```

The identification of the language requires the ```langid``` python module.

The parsing script needs a particular folder structure:

```
    +-- photos
    |   +-- AbuGhraib
            +-- AbuGhraib_source
                +-- AbuGhraib.jpg
            +-- AbuGhraib_1
                +-- AbuGhraib-gcv-output-name.json
                +-- img
                    +-- republication_image_1.jpg
                    +-- republication_image_2.jpg
            +-- AbuGhraib_2
                +-- republication_image_1_gcv_output.json
                +-- img
                    +-- republication_iter_2.jpg
    |   +-- AlanKurdi
            +-- AlanKurdi_source
            +-- AlanKurdi_1
            +-- AlanKurdi_2
```

After generating the .tsv files with ```transform2csv.py``` .html pages, texts and dates can be parsed with ```parse-csv.py```. 

Because our original pipeline included some noise in the results, we check if the original iconic image is present in some way on the online republished version. We use SIFT to do so. ```sift-image-checker.py``` downloads the image based on the links in the .tsv, preprocesses the image (by applying grayscale, downsampling and gaussian blur) and compares the original preprocessed image with the online republication.
