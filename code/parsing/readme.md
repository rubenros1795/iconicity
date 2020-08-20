# Parsing

This subfolder contains several scripts used from parsing data from the Google Cloud Vision API output (.json files) to .tsv format. Start with ```transform2csv.py``` , which parses data into a .tsv with the following columns:

```filename photo page_url image_url_full image_url_partial page_title iteration language labels scrape_date```

The identification of the language requires the ```langid``` python module.

The parsing script needs a particular folder structure:

```
​```
    +-- photo_folder
    |   +-- example_photo_1_folder
            +-- example_photo_1_folder_source
                +-- example_photo_1.jpg
    |   +-- example_photo_2_folder
            +-- example_photo_2_folder_source
                +-- example_photo_2.jpg
    ```
```