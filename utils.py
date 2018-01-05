import xmltodict
import pandas as pd
import os
import json
import time
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
     outdir = os.path.join('./out', df[:1].Congress[1])
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
    (rouge1_recall, rouge1_precision, rouge1_f1_score) = rouge_n(algo_summ, CRS_summ, n=1)
    (rouge2_recall, rouge2_precision, rouge2_f1_score) = rouge_n(algo_summ, CRS_summ, n=2)
    (r_lcs, p_lcs, f_lcs) = rouge_l_sentence_level(algo_summ, CRS_summ)

    rouge_dict = {'rouge1_R': rouge1_recall, 'rouge1_P': rouge1_precision, 'rouge1_F': rouge1_f1_score,
                'rouge2_R': rouge2_recall, 'rouge2_P': rouge2_precision, 'rouge2_F': rouge2_f1_score,
                'rougeL_R': r_lcs, 'rougeL_P': p_lcs, 'rougeL_F': f_lcs}

    return rouge_dict

def rouge_n(evaluated_sentences, reference_sentences, n=2):
    """
    Computes ROUGE-N of two text collections of sentences.
    Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
    papers/rouge-working-note-v1.3.1.pdf
    :param evaluated_sentences:
        The sentences that have been picked by the summarizer
    :param reference_sentences:
        The sentences from the referene set
    :param n: Size of ngram.  Defaults to 2.
    :returns:
        float 0 <= ROUGE-N <= 1, where 0 means no overlap and 1 means
        exactly the same.
    :raises ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise (ValueError("Collections must contain at least 1 sentence."))

    evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
    reference_ngrams = _get_word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    # Gets the overlapping ngrams between evaluated and reference
    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    recall = overlapping_count / reference_count
    precision = overlapping_count / evaluated_count
    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * ((precision * recall) / (precision + recall))

    return (recall, precision, f1_score)

def _get_word_ngrams(n, sentences):
    assert (len(sentences) > 0)
    assert (n > 0)

    words = set()
    for sentence in sentences:
        words.update(_get_ngrams(n, _split_into_words([sentence])))

    return words

def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _split_into_words(sentences):
    full_text_words = []
    for s in sentences:
        if not isinstance(s, Sentence):
            raise (ValueError("Object in collection must be of type Sentence"))
        full_text_words.extend(s.words)
    return full_text_words

def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
    """
    Computes ROUGE-L (sentence level) of two text collections of sentences.
    http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf
    Calculated according to:
    R_lcs = LCS(X,Y)/m
    P_lcs = LCS(X,Y)/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
    where:
    X = reference summary
    Y = Candidate summary
    m = length of reference summary
    n = length of candidate summary
    :param evaluated_sentences:
        The sentences that have been picked by the summarizer
    :param reference_sentences:
        The sentences from the referene set
    :returns float: F_lcs
    :raises ValueError: raises exception if a param has len <= 0
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise (ValueError("Collections must contain at least 1 sentence."))
    reference_words = _split_into_words(reference_sentences)
    evaluated_words = _split_into_words(evaluated_sentences)
    m = len(reference_words)
    n = len(evaluated_words)
    # if m == 0 or n == 0:
    #     return None
    # else:
    lcs = _len_lcs(evaluated_words, reference_words)
    (r_lcs, p_lcs, f_lcs) = _f_lcs(lcs, m, n)
    return (r_lcs, p_lcs, f_lcs)

def _len_lcs(x, y):
    """
    Returns the length of the Longest Common Subsequence between sequences x
    and y.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    :param x: sequence of words
    :param y: sequence of words
    :returns integer: Length of LCS between x and y
    """
    table = _lcs(x, y)
    n, m = _get_index_of_lcs(x, y)
    return table[n, m]

def _f_lcs(llcs, m, n):
    """
    Computes the LCS-based F-measure score
    Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
    rouge-working-note-v1.3.1.pdf
    :param llcs: Length of LCS
    :param m: number of words in reference summary
    :param n: number of words in candidate summary
    :returns float: LCS-based F-measure score
    """
    # try:
    r_lcs = llcs / m
    p_lcs = llcs / n
    if r_lcs == 0:
        f_lcs = 0
    else:
        beta = p_lcs / r_lcs
        num = (1 + (beta ** 2)) * r_lcs * p_lcs
        denom = r_lcs + ((beta ** 2) * p_lcs)
        f_lcs = num / denom
    # except:
    #     print(llcs, m, n)
    return (r_lcs, p_lcs, f_lcs)

def _lcs(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    :param x: collection of words
    :param y: collection of words
    :returns table: dictionary of coord and len lcs
    """
    n, m = _get_index_of_lcs(x, y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table

def _get_index_of_lcs(x, y):
    return len(x), len(y)



