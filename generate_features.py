'''
Before running this script, run the filter_summarize.ipynb to get 
./budget_only.tsv
'''
import sys
import os
import argparse
import pandas as pd
import string
import ipdb
from os import listdir, path
import cPickle as pickle
from os.path import isfile, join
from collections import defaultdict
from nltk.corpus import stopwords

from utils import *
import nltk
import string
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import cPickle as pickle
from collections import defaultdict
stemmer = nltk.PorterStemmer()



_SEP_ = '<FEATURE>'
_TITLE_SEP_ = '[]'
_IMPACT_WORDS = set(['cost', 'amount', 'import','$',"profit","loss"])
_MAX_FREQ_ = 1000
_MIN_FREQ_ = 3
TOKEN_FREQ_FILE = "bill_title_tokens.pkl"
def getFiles(folder, budget_only_suffix):
    billFiles = []
    summaryFiles = []
    titleFiles = []
    for f in listdir(folder):
        if isfile(join(folder, f)):
            suffix = f.split("_")
            suffix = suffix[2]+"_"+suffix[3]
            if suffix[:-4] in budget_only_suffix:
                pass
            if "BILL" in f:
                billFiles.append(folder+f)
            elif "SUMMARY" in f:
                summaryFiles.append(folder+f)
            elif "TITLE" in f:
                titleFiles.append(folder+f)

    billFiles_subset = []
    summaryFiles_subset = []
    titleFiles_subset = []
    for t in titleFiles:
        b = t.replace("TITLE","BILL")
        s = t.replace("TITLE", "SUMMARY")
        if b in billFiles and s in summaryFiles:
            titleFiles_subset.append(t)
            summaryFiles_subset.append(s)
            billFiles_subset.append(b)

    return billFiles_subset, summaryFiles_subset, titleFiles_subset
def remove_punct(array):
    array = array.lower().split()
    result = []
    for word in array:
        newword = word.replace("(","").replace(")","").replace(".","")
        newword = newword.replace(",","").replace("\"","").replace(":","").replace(";","")
        result.append(newword)
    return result
# ' '.join(remove_punct(budget_only['title'])).lower().split()

def create_title_dict(titleFiles, token_list):

    titleToFileCanonical = defaultdict(list)
    # find most important words
    titleToFileOriginal = defaultdict(list)
    for titleFile in titleFiles:
        with open(titleFile, "r") as f:
            
            title = f.readlines()
            title = title[0]
            # ipdb.set_trace()
            valid = []
            mtitle = remove_punct(title)
            for token in token_list:
                if token in mtitle: # treating a token as a word
                    valid.append(token)
        
        titleToFileCanonical[tuple(valid)].append((titleFile, title))
        titleToFileOriginal[tuple(mtitle)].append((titleFile,title))
    # print freqCount
    return titleToFileCanonical, titleToFileOriginal

def convertFileName(filename, orig, final):
    return filename.replace(orig, final)

def get_closest_title(mode_line, token_list, titleToFileCanonical):
    _max = float('-inf')
    _max_tup = None

    normal = 0

    for v in token_list:
        if v in mode_line:
            normal+=1
    if normal==0:
        return _max_tup, _max
    for tup in titleToFileCanonical:
        if len(tup)==1 and len(mode_line)>3:
            continue
        count = 0
        for w in tup:
            # ipdb.set_trace()
            if w in mode_line:
                count+=1
        
        match_score = 1.0*(count*count) / ((normal)*(len(tup)))
        if match_score> _max and match_score>0.8:
            _max = match_score
            _max_tup = tup
    return _max_tup, _max
def get_and_save_features(billFiles, titleFiles,token_list,out_folder, in_folder):
    titleToFileCanonical, titleToFileOriginal = create_title_dict(titleFiles, token_list)
    flag = 0
    countTitle = 0
    countImpact = 0
    proc = 0

    for billFile in billFiles:
        with open(billFile, "r") as f:
            text = ""
            for line in f:
                titleFeature = ""
                impactFeature = ""
                mode_line = remove_punct(line)
                if tuple(mode_line) in titleToFileCanonical:
                    print "Exact Match"
                else:
                    # impact Feature
                    for t in _IMPACT_WORDS:
                        if t in mode_line:
                            impactFeature+= "("+t+","+str(mode_line.count(t))+"),"
                    
                    if impactFeature!="":
                        impactFeature = impactFeature[:-1]

                    # titleFeature
                    closest, score = get_closest_title(mode_line, token_list, titleToFileCanonical)
                    if closest!=None:
                        # print closest
                        textOnly = []
                        filenames = []
                        for temp in titleToFileCanonical[closest]:
                            textOnly.append(temp[1])

                            tempFile = temp[0].replace(in_folder, out_folder)
                            
                            filenames.append(tempFile)
                        titleFeature = _TITLE_SEP_.join(filenames)
                        if len(filenames)>1:
                            print line,textOnly
                            raw_input()
                text+= line.strip("\n") + _SEP_+titleFeature+_SEP_+impactFeature+"\n"
                if titleFeature!="":
                    countTitle+=1
                if impactFeature!="":
                    countImpact+=1
                    # ipdb.set_trace()
            text = text.strip()
            outputFile = billFile.replace(in_folder, out_folder)
            with open(outputFile, "w") as f:
                f.write(text)
            if proc%100 == 0:
                print "Progress ", 100.0*proc/len(billFiles)
            proc+=1
    print "Impact Lines", countImpact
    print "Title matching lines", countTitle


def main(in_folder, subset_filename, out_folder):
    

    try:
        os.stat(out_folder)
    except:
        os.mkdir(out_folder)


    budget_only = pd.read_table(subset_filename)
    
    if os.path.isfile(TOKEN_FREQ_FILE):
        print "using old bill_title_tokens file"
        with open(TOKEN_FREQ_FILE,'r') as f:
            token_frequency = pickle.load(f)
    else:
        print "generating bill_title_tokens file for first time"
        title_tokens = pd.Series(' '.join(remove_punct(budget_only['title'])).lower().split()).value_counts()
        token_frequency = defaultdict(int)
        for k, v in title_tokens.iteritems():
            token_frequency[k] += v
        with open(TOKEN_FREQ_FILE, "w") as f:
            pickle.dump(token_frequency, f) 

    token_frequency = { k:v for k, v in token_frequency.items() if v  <= _MAX_FREQ_ and v>=_MIN_FREQ_ and k not in set(stopwords.words('english'))}
    token_list = set(token_frequency.keys())


    budget_only_suffix = set()
    for index, row in budget_only.iterrows():  # ensures we consider only budget only suffice
        budget_only_suffix.add(row["Number"]+"_"+row["Version"])
    assert len(budget_only) == len(budget_only_suffix)
    allTitles = budget_only['title']
    billFiles, summaryFiles, titleFiles = getFiles(in_folder, budget_only_suffix)
    
    assert len(billFiles) == len(summaryFiles) == len(titleFiles)
    print "Number of budget_file in ", in_folder, " are ", len(titleFiles)

    get_and_save_features(billFiles, titleFiles,token_list,out_folder, in_folder)


    

if __name__ == '__main__':
    try:
        parser=argparse.ArgumentParser(description = 'Script to extract Features for data')
        parser.add_argument('--out', '-o', dest = 'in_folder', help = 'path to out folder that holds bill, summary and title files', default='./out3/', required = False)
        parser.add_argument('--bFile', '-b', dest = 'subset_filename', help = 'path to budget only tsv file', default="./budget_only.tsv",required = False)
        parser.add_argument('--location', '-l', dest = 'out_folder', help = 'Path where to store output', default="./out_all/", required = False)
        args = vars(parser.parse_args())
    except:
        print "Please specify required arguments"
        
main(**args)