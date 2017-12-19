import pandas as pd
#from utils import *
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
import os
#import xmltodict
import json
from datetime import datetime as dt
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
# from sumy.evaluation.rouge import rouge_n
from collections import namedtuple
from sumy.models.dom import Sentence
import re
import subprocess
import time
import nltk

def clean_bill(text):
    try:
        split = re.split(r'(?i)\sAN\sACT\s{3,}|\sA\sBIL[L]*\s{2,}|\sA*\s[A-Z]*\sRESOLUTION\s{3,}', text)
        if len(split) > 1:
            text1 = split[1]
        elif len(text.split('In the House of Representatives, U. S.,')) > 1:
            text1 = text.split('In the House of Representatives, U. S.,')[1]
        else:
            text1 = text.split('_______________________________________________________________________')[1]
        if len(re.split(r'\s{2,}Passed the (Senate |House of Representatives )', text1)) > 1:
            print('XXXXXXXXXXXXXXXXXXxx')
        text21 = re.split(r'\s{2,}Passed the (Senate |House of Representatives )', text1)[0] # remove words after 'Attest:'
        text22 = re.split(r'\s{2,}Attest:\s{2,}', text21)[0] # remove words after 'Attest:'
        text23 = re.split(r'\s{2,}(Union |House )*Calendar No\.\s\d+\s{2,}', text22)[0] #
        text24 = re.split(r'\s{2,}Speaker of the House of Representatives\.\s{2,}', text23)[0]
        text3 = re.sub(r'[ ][\n][ ]+', r' ', text24) # somtimes one sentence may be break into multiple lines
        text4 = re.sub(r'\<all\>\s*$', r'', text3) # remove <all> in the end
        text5 = re.sub(r'\s', r' ', text4) # replace all kinds of white space (\n\t ) with ' '
        text6 = text5.replace('--', ' -- ') # add axtra space around --
        text7 = re.sub(r',\s+SECTION 1.', r'. SECTION 1', text6) # replace ',' before section 1 with '.'
        text8 = re.sub(r'(?i)(\sSEC)(\.)(\s\d)(\s*\.)', r'\1TION\3 ', text7) # replace 'SEC. x.' with 'SECTION x '
#         text9 = re.sub(r'Calendar No[\d\w\s\.]+$', r'', text8) # remove extra Calendar No.... in the end
        text10 = text8.replace(';', '.') # convert clause into sentences
        text11 = text10.replace("''. ", " ")
        text12 = text11.replace('--', ' ')
        text13 = text12.replace('``', " ")
        text14 = text13.replace("''", " ")
        return text14.strip()
    except:
        print('!!! Error at cleaning text:')
        print(text)
        return ' '
       # raise

def extra_clean(s):
    END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',")"]
    s_new = s.strip().replace('``', '"').replace("''", '"').replace('-LRB-', '(').replace('-RRB-', ')')
    if s_new[-1] in END_TOKENS: return s_new
    return s_new + "."

def split_file(data):
    ## with clean_summary and clean_bill
    text_b = data
    
    out_b = os.path.join('out_b.out')
    tmp_b = os.path.join('input.txt')
    with open(tmp_b, 'w', encoding="utf8") as f:
        f.write(clean_bill(text_b))
    

    command = ['java', '-cp', "*", 
               '-mx5g', 'edu.stanford.nlp.process.DocumentPreprocessor', tmp_b]
    
    # BILL_115_hr1_ih.out holds bills processed by standford nlp
    with open(out_b, "w", encoding="utf8") as outfile:
        subprocess.call(command, stdout=outfile)

    with open(out_b, "r+", encoding="utf8") as f:
        text_b = f.read()
        text_new = text_b.replace('-LRB-', '(').replace('-RRB-', ')')
        #rewrite the file
        f.seek(0)
        f.write(text_new)
        f.truncate()

    os.remove(tmp_b)

    with open(out_b, 'r', encoding="utf8") as f:
        text_s = f.readlines()
      
    bill_lengths = len(text_s)
    cleaning_bill = '\n'.join([extra_clean(i) for i in text_s])
    cleaning_bill = ' '.join(cleaning_bill.split('\n'))
    
    return cleaning_bill, bill_lengths

def run_sumy(text, algo='KL', sent_count=6):
    # time0 = time.time()
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    # time1 = time.time()
    stemmer = Stemmer("english")
    # time2 = time.time()

    if algo == 'KL':
        summarizer = KLSummarizer(stemmer)
    elif algo == 'LexRank':
        summarizer = LexRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")
    # time3 = time.time()

    summary_list = summarizer(parser.document, sent_count)
    # time4 = time.time()

    # print('Parse time: {} \t Stem time: {} \t Stop words time: {} \t Summarizer time: {}'.format(time1, time2, time3, time4))

    return summary_list
