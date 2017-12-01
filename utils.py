import xmltodict
import pandas as pd
import os
import json
from datetime import datetime as dt
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.evaluation.rouge import rouge_n
from collections import namedtuple

LEGISLATION_MAP = {
    'HR': 'Bills',
    'S': 'Bills',
    'HJRES': 'Joint Resolutions',
    'SJRES': 'Joint Resolutions',
    'HCONRES': 'Concurrent Resolutions',
    'SCONRES': 'Concurrent Resolutions',
    'HRES': 'Simple Resolutions',
    'SRES': 'Simple Resolutions'
}

def walk_dirs(path):
    seen = set()
    for root, dirs, files in os.walk(path, topdown=False):
        if dirs:
            parent = root
            while parent:
                seen.add(parent)
                parent = os.path.dirname(parent)
        for d in dirs:
            d = os.path.join(root, d)

            if d not in seen:
                b_dir = d

                if len(d.split('/')) > 6:
                    b_version = d.split('/')[-1].upper()
                else:
                    b_version = 'N/A'
                b_subtype = d.split('/')[4].upper()
                b_type = LEGISLATION_MAP[b_subtype]
                b_number = d.split('/')[5].upper()

                summary_dir = '/'.join(x for x in d.rsplit('/')[:6]) + "/data.json"
                with open(summary_dir) as json_data:
                    j = json.load(json_data)
                if j['summary']:
                    b_summary = 1
                else:
                    b_summary = 0

                b_congress = d.split('/')[2]
                b_id = b_congress+'_'+b_number+'_'+b_version

                data_dict = {
                    'Directory': b_dir, 'Type': b_type, 'Subtype': b_subtype, 'Number': b_number,
                    'Version': b_version, 'Summary': b_summary, 'ID': b_id, 'Congress': b_congress
                }

                yield data_dict

def get_recent_bills(with_summary):
    for bill_no in with_summary.Number.unique():
        df = with_summary[with_summary.Number == bill_no][with_summary.Version != 'EAS']
        if len(df) > 1:
            if 'ENR' in list(df.Version):
                idx = with_summary[with_summary.Number == bill_no][with_summary.Version == 'ENR'].index
            else:
                all_date_info = []
                for folder_path in df.Directory:
                    date_info = {}
                    with open(folder_path+'/document.xml') as f:
                        dict1 = xmltodict.parse(f.read())
                    version = folder_path.split('/')[-1]
                    date_info['version'] = version
                    if 'bill' in dict1.keys():
                        bill = dict1['bill']
                        date = None
                        if 'form' in bill.keys():
                            form = bill['form']
                            if 'action' in form.keys():
                                action = form['action']
                                if isinstance(action, list):
                                    action_dict = action[0]
                                else:
                                    action_dict = action
                                if '@date' in action_dict.keys():
                                    date = action_dict['@date']
                                else:
                                    str_date = action_dict['action-date']
                                    if isinstance(str_date, str):
                                        date = dt.strftime(dt.strptime(str_date, '%B %d, %Y'), '%Y%m%d')
                                    elif isinstance(str_date, dict):
                                        date = str_date['@date']
                                    else:
                                        print('str date', str_date)
                        if date == None and 'attestation' in bill.keys():
                            date = bill['attestation']['attestation-group']['attestation-date']['@date']
                    if date == None:
                        print('here dict', folder_path, dict1, '\n')
                    if date != None:
                        date_info['date'] = date
                    else:
                        print(folder_path)
                    all_date_info.append(date_info)
                df2 = pd.DataFrame(all_date_info, index=range(len(all_date_info)))
                to_use = df2.sort_values('date', ascending=False)['version'][0].upper()
                idx = with_summary[with_summary.Number == bill_no][with_summary.Version == to_use].index                
        else:
            idx = with_summary[with_summary.Number == bill_no].index
        with_summary.loc[idx, 'to_use'] = 1
    return with_summary

def run_sumy(text, algo='KL', sent_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    stemmer = Stemmer("english")

    if algo == 'KL':
        summarizer = KLSummarizer(stemmer)
    elif algo == 'LexRank':
        summarizer = LexRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")

    summary_list = summarizer(parser.document, sent_count)

    return summary_list

def convert_summary(CRS_summ):
    parser = PlaintextParser.from_string(CRS_summ, Tokenizer("english"))
    rating = dict(zip(parser.document.sentences, [1 for i in parser.document.sentences]))
    SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))
    rate = lambda s: rating[s]
    infos = (SentenceInfo(s, o, rate(s)) for o, s in enumerate(parser.document.sentences))
    crs_tuple = tuple(i.sentence for i in infos)

    return crs_tuple

def eval_sumy(algo_summ, CRS_summ):
    rouge1 = rouge_n(algo_summ, CRS_summ, n=1)
    rouge2 = rouge_n(algo_summ, CRS_summ, n=2)

    return (rouge1, rouge2)







