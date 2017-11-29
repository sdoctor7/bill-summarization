# Code modified from bexcer's python3 version (https://github.com/becxer/cnn-dailymail/)
# which originally from https://github.com/abisee/pointer-generator

import sys
import os
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
import pandas as pd

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'


tokenized_files_dir = "files_tokenized"

finished_files_dir = "finished_files"
chunks_dir = os.path.join(finished_files_dir, "chunked")

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data


def chunk_file(set_name):
  in_file = 'finished_files/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % chunks_dir)


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.replace('-RRB-', ')').replace('-LRB-', '(').strip())
    return lines


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print(line[-1])
    return line + " ."


def get_bill(id):
    file_path = './tmp/out3/' + type + '_' + id + '.out'

    lines = read_text_file(file_path)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    if type == 'BILL':
        # Make article into a single string
        text = ' '.join(lines)
    elif type == 'SUMMARY':
        # Make abstract into a signle string, putting <s> and </s> tags around the sentences
        text = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in lines])
    else:
        sys.exit("Invalid file type. It should be BILL or SUMMARY.")

    return text


def get_data(id, type):
    file_path = os.path.join(data_dir, type+'_'+id+'.out')

    lines = read_text_file(file_path)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them (
    # this is a problem in the dataset because many image captions don't end in periods;
    # consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    if type == 'BILL':
        # Make article into a single string
        text = ' '.join(lines)
    elif type == 'SUMMARY':
        # Make abstract into a signle string, putting <s> and </s> tags around the sentences
        text = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in lines])
    else:
        sys.exit("Invalid file type. It should be BILL or SUMMARY.")

    return text


def tokenize_data(data_dir, tokenized_files_dir):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." % (data_dir, tokenized_files_dir))
    text = os.listdir(data_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in text:
            f.write("%s \t %s\n" % (os.path.join(data_dir, s), os.path.join(tokenized_files_dir, s)))
    # command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    command = ['java', '-classpath',
               "/home/lucy/stanford-corenlp/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar:/home/lucy/stanford-corenlp/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0-models.jar",
               'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']

    print("Tokenizing %i files in %s and saving in %s..." % (len(text), data_dir, tokenized_files_dir))
    subprocess.run(command)
    print("Stanford CoreNLP Tokenizer has finished.")

    os.remove("mapping.txt")

    # Check that the tokenized data directory contains the same number of files as the original directory
    num_orig = len(os.listdir(data_dir))
    num_tokenized = len(os.listdir(tokenized_files_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized data directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_files_dir, num_tokenized, data_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (data_dir, tokenized_files_dir))


def write_to_bin(df, out_file, makevocab=False):
    # """Reads the tokenized files and writes them to a out_file."""
    print("Making bin file for data ...")

    num_data = len(df)

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print("Writing data %i of %i; %.2f percent done" % (
                idx, num_data, float(idx) * 100.0 / float(num_data)))

            # Get the strings to write to .bin file
            bill = get_data(row.id, 'BILL')
            summary = get_data(row.id, 'SUMMARY')

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([bill.encode()])
            tf_example.features.feature['abstract'].bytes_list.value.extend([summary.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = bill.split(' ')
                abs_tokens = summary.split(' ')
                abs_tokens = [t for t in abs_tokens if
                              t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':

    if len(sys.argv) != 4:
      print("USAGE: python make_datafiles.py <data_dir> <train_list_dir> <validate_list_dir> <test_list_dir>")
      print("e.g. python make_datafiles.py './tmp/out3' './data/train.txt' './data/validate.txt' './data/test.txt'")
      sys.exit()
    # cnn_stories_dir = sys.argv[1]
    # dm_stories_dir = sys.argv[2]

    data_dir = sys.argv[1]
    train_dir = sys.argv[2]
    validate_dir = sys.argv[3]
    test_dir = sys.argv[4]

    # data_dir = './tmp/out3'
    # train_dir = './data/train.txt'
    # validate_dir = './data/validate.txt'
    # test_dir = './data/test.txt'


    # Check the stories directories contain the correct number of .story files
    # check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
    # check_num_stories(dm_stories_dir, num_expected_dm_stories)

    # Create some new directories
    if not os.path.exists(tokenized_files_dir): os.makedirs(tokenized_files_dir)
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
    tokenize_data(data_dir, tokenized_files_dir)

    df_train = pd.read_csv(train_dir)
    df_validate = pd.read_csv(validate_dir)
    df_test = pd.read_csv(test_dir)

    # Read the tokenized stories, do a little postprocessing then write to bin files
    write_to_bin(df_test, os.path.join(finished_files_dir, "test.bin"))
    write_to_bin(df_validate, os.path.join(finished_files_dir, "val.bin"))
    write_to_bin(df_train, os.path.join(finished_files_dir, "train.bin"), makevocab=True)

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing
    # e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all()
