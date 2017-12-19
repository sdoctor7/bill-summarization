# bill-summarization
Text Summarization of Congressional Bills: Fall 2017 Capstone project, Columbia University Data Science Institute and Bloomberg


<!-- # observations -->
<!-- Impact Lines 11711 -->
<!-- Title matching lines 2269 -->

<!-- # Initial Observations -->

<!-- | Observation | Count | -->
<!-- | ------ | ------ | -->
<!-- | Total Files Received | 8837 | -->
<!-- | Bills with valid file | 8759 | -->
<!-- | Bills with at least one summary | 5395 | -->
<!-- | Bills with at least two summary | 670 | -->
<!-- | Bills with at least three summary | 160 | -->
<!-- | Bills with at least four summary | 54 | -->


# Download Data

Data are collected from two resources:
1. Run app of https://github.com/unitedstates/congress
    - clone the repo
    - run command `./run fdsys --collections=BILLS --congress=114 --store=mods,xml,text --bulkdata=False`


2. Download *.zip from https://www.propublica.org/datastore/dataset/congressional-data-bulk-legislation-bills


# Process and split data

This part clean the bill and summary text and filtered the data we want.

- `filter_and_prepare_data.ipynb`
    1) Load, Deduplicate, Filter and Split data
        - Filter file with summary and bill text
        - For bills with more than one version, we picked the most recent one according to `document.xml`, if it not exists, just pick the first.
        - Output selected bill and summary files. One sentence per line.
        - Files stored under `./out/<congress_number>`
        - Files naming convention: `'BILL' + '_' + row['ID'].out` and `'SUMMARY' + '_' + row['ID'].out` (e.g. `BILL_113_HR1_IH.out`)


             | Congress | Count |
             | ------ | ------ |
             | 115 | 4114 |
             | 114 | 10045 |
             | 113 | 8903 |


    2) Split train, val, test


         | Train | Validation | Test | Total |
         | ------ | ------ | ------ | ------ |
         | 18449 | 2306 | 2307 | 23062 |


- `make_datafiles.py`

It processes data and creates two folders: 1) `file_tokenized` which we do not use; 2) `finished_files` which stores chucked *.bin files for pointer-generator.

      - USAGE: `python make_datafiles.py <data_dir> <train_list_dir> <validate_list_dir> <test_list_dir>`
      - `e.g. python make_datafiles.py './out/113_114_115' './out/train_113_114_115.txt' './out/validate_113_114_115.txt' './out/test.txt_113_114_115'`

- Code modified from bexcer's python3 version (https://github.com/becxer/cnn-dailymail/) which originally from abisee (https://github.com/abisee/pointer-generator)


# Extractive Summarizer (sumy)



# Abstractive Summarizer (Pointer Generator)

Details can be found at
- http://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html
- https://github.com/becxer/pointer-generator/


##### Train
~~~
python run_summarization.py --mode=train --data_path="/home/lucy/Workspace/bill-summarization/finished_files/chunked/train_*" --vocab_path="/home/lucy/Workspace/bill-summarization/finished_files/vocab" --log_root="/home/lucy/Workspace/pointer-generator/log" --exp_name=bill-113-114-115 --batch_size=32
~~~

##### Evaluation (concurrent)
~~~
python run_summarization.py --mode=eval --data_path="/home/lucy/Workspace/bill-summarization/finished_files/chunked/val_*" --vocab_path="/home/lucy/Workspace/bill-summarization/finished_files/vocab" --log_root="/home/lucy/Workspace/pointer-generator/log" --exp_name=bill-113-114-115
~~~


##### Decoding

1) validation data (run all file)
~~~
python run_summarization.py --mode=decode --data_path="/home/lucy/Workspace/bill-summarization/finished_files/chunked/val_*" --vocab_path="/home/lucy/Workspace/bill-summarization/finished_files/vocab" --log_root="/home/lucy/Workspace/pointer-generator/log" --exp_name=bill-113-114-115 --single_pass=1
~~~

2) validation data (produce one attn_vis_data.json file for the attention visualizer)
~~~
python run_summarization.py --mode=decode --data_path="/home/lucy/Workspace/bill-summarization/finished_files/chunked/val_*" --vocab_path="/home/lucy/Workspace/bill-summarization/finished_files/vocab" --log_root="/home/lucy/Workspace/pointer-generator/log" --exp_name=bill-113-114-115
~~~


3) test data (run all file)
~~~
python run_summarization.py --mode=decode --data_path="/home/lucy/Workspace/bill-summarization/finished_files/chunked/test_*" --vocab_path="/home/lucy/Workspace/bill-summarization/finished_files/vocab" --log_root="/home/lucy/Workspace/pointer-generator/log" --exp_name=bill-113-114-115 --single_pass=1
~~~

4) test data (produce one attn_vis_data.json file for the attention visualizer)
~~~
python run_summarization.py --mode=decode --data_path="/home/lucy/Workspace/bill-summarization/finished_files/chunked/test_*" --vocab_path="/home/lucy/Workspace/bill-summarization/finished_files/vocab" --log_root="/home/lucy/Workspace/pointer-generator/log" --exp_name=bill-113-114-115
~~~

##### Visualize one output

- Clone: https://github.com/abisee/attn_vis
- Move attn_vis_data.json from log to the repository
- Run `python -m SimpleHTTPServer` (python2 only)



#  Find Budget-related Bills
- `filter_summarize.ipynb`


# Feature


# Analysis

# User Interface
- Link: http://35.196.17.188:8111/   
- Introduction: The web application is built as a Flask Python web server and deployed to the Google App Engine virtual machine.


