import xmltodict
import pandas as pd
import os
import re
import json
from datetime import datetime as dt
import subprocess
import concurrent.futures

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
                try:
                    with open(summary_dir) as json_data:
                        j = json.load(json_data)
                    if j['summary']:
                        b_summary = 1
                    else:
                        b_summary = 0
                except FileNotFoundError:
                    print('No data.json file: {}'.format(summary_dir))
                    b_summary = 0

                b_congress = d.split('/')[2]
                b_id = b_congress+'_'+b_number+'_'+b_version

                data_dict = {
                    'Directory': b_dir, 'Type': b_type, 'Subtype': b_subtype, 'Number': b_number,
                    'Version': b_version, 'Summary': b_summary, 'ID': b_id, 'Congress': b_congress
                }

                yield data_dict


def load_data(congress_number, hrsonly=None):
    # Bills
    INDIR_HR = './data/'+congress_number+'/bills/hr'
    INDIR_S = './data/'+congress_number+'/bills/s'

    # Concurrent Resolutions
    INDIR_HCONRES = './data/'+congress_number+'/bills/hconres'
    INDIR_SCONRES = './data/'+congress_number+'/bills/sconres'

    # Joint Resolutions
    INDIR_HJRES = './data/'+congress_number+'/bills/hjres'
    INDIR_SJRES = './data/'+congress_number+'/bills/sjres'

    # Simple Resolutions
    INDIR_HRES = './data/'+congress_number+'/bills/hres'
    INDIR_SRES = './data/'+congress_number+'/bills/sres'

    if hrsonly:
        indirs = [INDIR_HR, INDIR_S]
        print('Only Load HR and S')
    else:
        indirs = [INDIR_HR, INDIR_S,
                     INDIR_HCONRES, INDIR_SCONRES,
                     INDIR_HJRES, INDIR_SJRES,
                     INDIR_HRES, INDIR_SRES
           ]

    data = []

    for i in indirs:
        print('Processing {}'.format(i))
        for d in walk_dirs(i):
            data.append(d)

    df = pd.DataFrame(data)
    print('Number of rows: {}'.format(len(df)))
    print('Number of unique bills: {}'.format(len(df.Number.unique())))

    return df


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


def dedup_filter_bills(df):
    print('Remove bills with no bill text: {}'.format(len(df[df.Version == 'N/A'])))
    with_bill_text = df[df.Version != 'N/A'].copy()
    # print('Number bills with bill texts: {}'.format(len(with_bill_text)))
    print('Remove bills with no summary: {}'.format(len(with_bill_text[with_bill_text.Summary == 0])))
    with_summary = with_bill_text[with_bill_text.Summary > 0].copy()
    # print('Number of bills left: {}'.format(len(with_summary)))

    # print('Number of unique bills with bill text: {}'.format(len(with_bill_text.Number.unique())))
    print('Getting most recent version for duplicated bills')
    with_summary['to_use'] = 0
    recents_marked = get_recent_bills(with_summary)
    unique_bills = recents_marked[recents_marked.to_use == 1]
    print('Number of unique bills: {}'.format(len(unique_bills)))
    return unique_bills


def clean_bill(text):
    try:
        split = re.split(r'(?i)\sAN\sACT\s{3,}|\sA\sBIL[L]*\s{2,}|\sA*\s[A-Z]*\sRESOLUTION\s{3,}', text)
        if len(split) > 1:
            text1 = split[1]
        elif len(text.split('In the House of Representatives, U. S.,')) > 1:
            text1 = text.split('In the House of Representatives, U. S.,')[1]
        else:
            text1 = text.split('_______________________________________________________________________')[1]
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
        text10 = text8.replace(';', '.')
        text11 = text10.replace("''. ", " ")
        text12 = text11.replace('--', ' ')
        text13 = text12.replace('``', " ")
        text14 = text13.replace("''", " ")
        return text14.strip()
    except:
        print('!!! Error at cleaning text:')
        print(text)
#         return ' '
        raise


def clean_summary(text):
    summary = ''
    text1 = re.sub(r'\(This measure has not been amended[\w\s\d,\.]+\)', ' ', text)
    for s in text1.split('\n'):
        if s.strip().rstrip():
            summary += s+'\n'
    return summary


def split_file(data, cp='/home/lucy/stanford-corenlp/stanford-corenlp-full-2016-10-31/'):
    i, row = data
    outdir = os.path.join('out', row['Congress'])

    indir_b = os.path.join(row['Directory'], 'document.txt')

    filename_b = 'BILL' + '_' + row['ID']
    tmp_b = os.path.join(outdir, filename_b)
    out_b = os.path.join(outdir, filename_b + '.out')

    with open(indir_b, 'r') as f:
        text_b = f.read()

    with open(tmp_b, 'w') as f:
        f.write(clean_bill(text_b))

    command = ['java', "-classpath",
               cp + 'stanford-corenlp-3.7.0.jar:' + cp + 'stanford-corenlp-3.7.0-models.jar',
               'edu.stanford.nlp.process.DocumentPreprocessor', tmp_b]

    with open(out_b, "w") as outfile:
        subprocess.run(command, stdout=outfile)

    with open(out_b, "r+") as f:
        text_b = f.read()
        text_new = text_b.replace('-LRB-', '(').replace('-RRB-', ')')
        f.seek(0)
        f.write(text_new)
        f.truncate()

    os.remove(tmp_b)

    indir_s = os.path.join(row['Directory'], '..', '..', 'data.json')
    filename_s = 'SUMMARY' + '_' + row['ID']
    # tmp_s = os.path.join(outdir, filename_s)
    out_s = os.path.join(outdir, filename_s + '.out')

    # split summary
    with open(indir_s, 'r') as f:
        text_s = json.load(f)['summary']['text']

    with open(out_s, 'w') as f:
        f.write(clean_summary(text_s))


def concurrent_split(df):
    counts = range(1, len(df)+1)
    outdir = os.path.join('out', df.Congress[0])
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, _ in zip(counts, executor.map(split_file, df.iterrows())):
            if i % 100 == 0:
                print('Spliting file: {}/{} bills'.format(i, len(df)))

                
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


