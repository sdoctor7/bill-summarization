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

import time
def run_sumy(text, algo='KL', sent_count=3):
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

    rouge_dict = {'rouge1_recall': rouge1_recall, 'rouge1_precision': rouge1_precision, 'rouge1_f': rouge1_f1_score,
                'rouge2_recall': rouge2_recall, 'rouge2_precision': rouge2_precision, 'rouge2_f': rouge2_f1_score,
                'rougeL_recall': r_lcs, 'rougeL_precision': p_lcs, 'rougeL_F': f_lcs}

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



