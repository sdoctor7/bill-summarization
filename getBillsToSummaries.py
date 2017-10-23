import xmltodict
import pandas as pd
import os
import re
billDir = 'bill_text_115' 
summariesDir = '../data/summaries/'

def getBillFileName(summaryFileName, extension):
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

def print_head(d,top=10):
    count = 0
    for k in d:
        if count==top:
            break
        print ("{} -> {}".format(k,d[k]))
        count+=1
    return

def getBillsToSummaries(billDir='bill_text_115', summariesDir='../data/summaries/'):

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
#     print_head(billsToSummary)
    return billsToSummary, billFiles, summariesNoMatch