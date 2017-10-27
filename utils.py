import xmltodict
import pandas as pd
import os
import re
from nltk.tokenize import sent_tokenize, word_tokenize

def remove_tags(read_file):
    read_file = re.sub(b'</?external-xref[^<>]*>',b'', read_file)
    read_file = re.sub(b'<quote>',b'"', read_file)
    read_file = re.sub(b'</quote>',b'"', read_file)
    read_file = re.sub(b'</?term[^<>]*>',b' ', read_file)
    read_file = re.sub(b'</?pagebreak[^<>]*>',b'', read_file)
    return read_file

def flatten(d):
    output = ''
    for k, v in d.items():
        if not k.startswith("@"):
            if isinstance(v, dict):
                output += flatten(v)
            elif isinstance(v, list):
                for l in v:
                    if isinstance(l, dict):
                        output += flatten(l)
                    else:
                        if l:
                            output += l + ' '
            else:
                if v:
                    output += v + ' '
    return output

def section_bill(d):
    global section
    if isinstance(d, dict):
        if 'section' in d.keys():
            section += 1
        if 'subsection' in d.keys():
            section += 1
        if 'paragraph' in d.keys():
            section += len(d['paragraph'])
        for k, v in d.items():
            if not k.startswith("@"):
                section_bill(v)
    elif isinstance(d, list):
        for l in d:
            section_bill(l)
    else:
        if d:
            section = section
    return section

def length_bill(text):
    word = word_tokenize(text)
    word = len([s for s in word if re.match(r'.*[A-Za-z0-9].*',s)])
    # Use period to detect sentence or semicolon?
    sent_tokenize_list = sent_tokenize(text)
    sentence = len(sent_tokenize_list)
    return sentence, word

def bill_to_dict(filename, doc):
    def _clean_body(bodydict):
        clean_bodydict = {}
        for k, v in bodydict.items():
            if not k.startswith("@"):
                if isinstance(v, list):
                    sec_str = ''
                    for sub_dict in v:
                        if sub_dict:
                            sec_str += flatten(sub_dict)
                    clean_bodydict[k] = sec_str
                else:
                    clean_bodydict[k] = flatten(v)
        return clean_bodydict
    
    bill_type = list(doc)[0]
    data_dict = {}
    
    data_dict["file-name"] = filename
    data_dict["bill-type"] = bill_type
    data_dict['official-title'] = None
    data_dict['legis-type'] = None
    data_dict['dc:title'] = None
    
    metadata = {}
    for n in list(doc[bill_type]):
        # group meta data
        if n.startswith("@"): 
            metadata[n] = doc[bill_type][n]
        
        # unify name for different types
        elif n in ['legis-body', 'resolution-body', 'engrossed-amendment-body']:
            if isinstance(doc[bill_type][n], dict):
                data_dict['body'] = _clean_body(doc[bill_type][n])  
                data_dict['whole_body'] = flatten(doc[bill_type][n])
            elif isinstance(doc[bill_type][n], list):
                # just take last one
                data_dict['body'] = _clean_body(doc[bill_type][n][-1])
                data_dict['whole_body'] = flatten(doc[bill_type][n][-1])
            else:
                print('NOT dict nor list')
                data_dict['body'] = doc[bill_type][n]
            
            ## calculate the length of bills in paragraphs, sentences and words
            global section
            section = 0
            if isinstance(doc[bill_type][n], dict):
                section = section_bill(doc[bill_type][n])
            elif isinstance(doc[bill_type][n], list):
                section = section_bill(doc[bill_type][n][-1])
            else:
                section = 0
            data_dict['section'] = section
            try:
                sentence, word = length_bill(data_dict['whole_body'])
            except Exception as e:         
                print("While counting length, the error occurs: {}".format(e))
            finally:
                data_dict['sentence'] = sentence
                data_dict['word'] = word
                
        elif n == 'engrossed-amendment-form':
            data_dict['form'] = doc[bill_type][n]
            
        ## add fields legis-type, official-title from 'form'
        elif n == 'form':
            try:
                data_dict['legis-type'] = doc[bill_type][n]['legis-type']
                data_dict['official-title'] = doc[bill_type][n]['official-title']
                if isinstance(data_dict['official-title'], dict):
                    data_dict['official-title'] = data_dict['official-title']['#text']
                data_dict['official-title'] = re.compile(r'[\n\r\t]').sub("", data_dict['official-title'])
                if isinstance(data_dict['legis-type'], dict):
                    data_dict['legis-type'] = data_dict['legis-type']['#text']
            except Exception as e:
                print("Do not exist %s"%e)
                if e == '#text':
                    data_dict['official-title'] = None
                    
        ## add field dc:title from 'metadata'
        elif n == 'metadata':
            try:
                data_dict['dc:title'] = doc[bill_type][n]['dublinCore']['dc:title']
            except Exception as e:
                print("Do not exist %s"%e)
                data_dict['dc:title'] = None
        else:
            data_dict[n] = doc[bill_type][n]

    data_dict["metadata"] = metadata
    return data_dict

def parse_summary(path, fileName):

    with open(path+fileName, 'rb') as file:
        dict1 = xmltodict.parse(file.read()) # parse original XML to a dictionary
    
    if 'billStatus' in dict1.keys():
        
        dict2 = {} # initialize empty dictionary for this bill
        dict2['fileName'] = fileName # insert filename
        dict2['billNumber'] = dict1['billStatus']['bill']['billNumber'] # insert bill number
        dict2['contributor'] = dict1['billStatus']['dublinCore']['dc:contributor'] # insert contributor

        ### summaries (there may be multiple) ###
        summaries = dict1['billStatus']['bill']['summaries']['billSummaries']
        if summaries:
            if isinstance(summaries['item'], dict): # if there's only one summary
                # remove HTML tags from the summary and append it
                dict2['summary0'] = BeautifulSoup(summaries['item']['text'], 'lxml').text
                # add length fields (sentence and word) to columns
                dict2['sentence0'], dict2['word0'] = length_bill(dict2['summary0'])
            elif isinstance(summaries['item'], list): # if there are multiple summaries
                for i, item in enumerate(summaries['item']):
                    # remove HTML tags from each summary and append it
                    dict2['summary'+str(i)] = BeautifulSoup(item['text'], 'lxml').text
                    # add length fields (sentence and word) to columns
                    dict2['sentence'+str(i)], dict2['word'+str(i)] = length_bill(dict2['summary'+str(i)])
        
        ### titles (there may be multiple) ###
        dict2['title'] = dict1['billStatus']['bill']['title']
        titles = dict1['billStatus']['bill']['titles']['item'] # original title in 'title' tag
        for i, item in enumerate(titles): # all other titles
            dict3 = {}
            dict3[item['titleType']] = item['title']
            dict2['title'+str(i)] = str(dict3)

        return (1, dict2)
    
    else:
        return (0, fileName)

def getBillFileName(summaryFileName, extension):
    """
    Return bill file names based on summaryfilenames
    """
#     US_Bill_Text_115_HR1607_IH.xml <- US_Bill_Digest_115_hr_1607.xml
#     possibleExt = ["_RH.xml","_IH.xml","_EH.xml","_RFS.xml","_IS.xml"]
    billFileName = []
    if summaryFileName.startswith("US_Bill_Digest_115_"):
        tempFile = summaryFileName.split("_")
        tempFile = tempFile[-2:]
        tempFile[0] = tempFile[0].upper()
        for ext in extension:
            billFileName.append("US_Bill_Text_115_"+tempFile[0]+tempFile[1][:-4]+"_"+ext)
    return billFileName

def getBillsToSummaries(billDir, summariesDir):

    billFiles = set()
    extension = set()
    for root, dirs, filenames in os.walk(billDir):
        for filename in filenames:
            billFiles.add(filename)
            extension.add(filename.split("_")[-1])
    assert len(billFiles)==8039
    
    billsToSummary = {}
    summariesNoMatch = set()
    for root, dirs, filenames in os.walk(summariesDir):
        for filename in filenames:
            billFileList = getBillFileName(filename, extension)
            flag = 1
            for billF in billFileList:
                if billF in billFiles:
                    billsToSummary[billF] = filename
                    flag=0
            if flag:
                summariesNoMatch.add(filename)
    return billsToSummary, billFiles, summariesNoMatch