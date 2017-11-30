'''
Script to convert folder out that holds budget_only files 
and use train, text, validate.txt and put them into respective folders
'''



import os
import sys
from os import listdir
from os.path import isfile, join
TRAIN = "./train.txt"
TEST = "./test.txt"
VALIDATE = "./validate.txt"
FOLDER = "./out_bill_only"


def read_file_list(filename):
    index = 1
    file_path = set()
    with open(filename, "r") as f:
        for line in f:
            if index:
                index = 0
                continue
            temp = line.strip().split(",")
            #115,HCONRES35,./data/115/bills/hconres/hconres35/text-versions/rh,115_HCONRES35_RH,HCONRES,1,Concurrent Resolutions,RH
            new_path = FOLDER+"/BILL_"+temp[0]+"_"+temp[1]+"_"+temp[-1]+".out"
            file_path.add(new_path)
    return file_path

def ensure_dir(dirname):
    try:
        os.stat(dirname)
    except:
        os.mkdir(dirname)

def moveFiles(filenames, folder):
    print filenames
    for f in filenames:
        newPath = f.replace(FOLDER, folder)
        # print f, folder
        os.rename(f,newPath)
def split_dataset():
    train_list_complete = read_file_list(TRAIN)
    test_list_complete = read_file_list(TEST)
    validate_list_complete = read_file_list(VALIDATE)

    train_list_budgetType = []    # stores the actual bill
    test_list_budgetType = []
    validate_list_budgetType = []

    for f in listdir(FOLDER):
        filename = join(FOLDER, f)
        if isfile(filename):
            if filename in train_list_complete:
                train_list_budgetType.append(filename)
            elif filename in validate_list_complete:
                validate_list_budgetType.append(filename)
            elif filename in test_list_complete:
                test_list_budgetType.append(filename)
    # 
    train_folder = join(FOLDER, "train")
    test_folder = join(FOLDER,"test")
    val_folder = join(FOLDER, "val")
    ensure_dir(train_folder)
    ensure_dir(test_folder)
    ensure_dir(val_folder)
    
    moveFiles(train_list_budgetType, train_folder)
    moveFiles(test_list_budgetType, test_folder)
    moveFiles(validate_list_budgetType, val_folder)


    # raw_input()

if __name__ == '__main__':

    split_dataset()