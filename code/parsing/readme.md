# Parsing

This subfolder contains several scripts used from parsing data from the Google Cloud Vision API output (.json files) to .tsv format. Start with ```transform2csv.py``` , which parses data into a .tsv with the following columns:

```filename photo page_url image_url_full image_url_partial page_title iteration language labels scrape_date```

The identification of the language requires the ```langid``` python module.

The parsing script needs a particular folder structure:

```
    +-- photo_folder
    |   +-- example_photo_1_folder
            +-- example_photo_1_folder_source
                +-- example_photo_1.jpg
    |   +-- example_photo_2_folder
            +-- example_photo_2_folder_source
                +-- example_photo_2.jpg
```

After generating the .tsv files with ```transform2csv.py``` dates can be scraped with ```parse-csv-dates.py```. Similarly, .html pages are scraped and parsed for textual content with ```parse-csv-html.py``` and ```parse-csv-text.py```.

Because our original pipeline included some noise in the results, we check if the original iconic image is present in some way on the online republished version. We use SIFT to do so. ```sift-image-checker.py``` downloads the image based on the links in the .tsv, preprocesses the image (by applying grayscale, downsampling and gaussian blur) and compares the original preprocessed image with the online republication.
