from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.kl import KLSummarizer

from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


def run_sumy(text, algo='KL', sent_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    stemmer = Stemmer("english")

    if algo == 'KL':
        summarizer = KLSummarizer(stemmer)
    elif algo == 'LexRank':
        summarizer = LexRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")

    summary = ''
    for sentence in summarizer(parser.document, sent_count):
        summary += str(sentence) + ' '
    return summary


