'''
Script accepts bills in a folder, with feature as decided by get_features
Creates a graph of bills related to each other and also for impact words
'''
import os
import networkx as nx
import sys
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import ipdb
import numpy as np
FOLDER = "./out_bill_only"
_SEP_ = '<FEATURE>'
_TITLE_SEP_ = '[]'
_IMPACT_WORDS = ['cost', 'amount', 'import','$']


def get_node_name(filename):
    # BILL_115_HCONRES2_IH.out
    name = filename.strip().split("/")[2].split("_")
    name = "_".join(name[2:])
    name = name[:-4]
    return name

def visualize():
    bill_related = defaultdict(int)
    bill_impact = defaultdict(list)
    impact_count = np.array([0]*len(_IMPACT_WORDS))

    for f in listdir(FOLDER):
        filename = join(FOLDER, f)
        if isfile(filename):
            src = get_node_name(filename)
            with open(filename,"r") as f:
                impact_vector = np.array([0]*len(_IMPACT_WORDS))
                count = 0
                for line in f:
                    sentence, related_bills, impact_word_string = line.strip().split(_SEP_)
                    related_bills = related_bills.split(_TITLE_SEP_)
                    if related_bills != ['']:
                        # print "related_words", related_bills
                        for dest_fileName in related_bills:
                            dest = get_node_name(dest_fileName)
                            if (dest, src) not in bill_related:
                                bill_related[(src,dest)]+=1
                            else:
                                bill_related[(dest,src)]+=1

                    if impact_word_string != '':
                        impact_word_list = impact_word_string.split(",")
                        for i in xrange(0, len(impact_word_list), 2):
                            impact_word = impact_word_list[i].strip("(")
                            impact_count = int(impact_word_list[i+1].strip(")"))
                            impact_vector[_IMPACT_WORDS.index(impact_word)]+=count
                    impact_count+=impact_vector
                    count+=1
                bill_impact[src] = 1.0* (impact_vector) / count
   
    # code to generate vertices and edges file for systemg
    vertices = ["Node_Name"]
    edges = ["Node_A, Node_B, Weight"]
    seen_nodes = set()
    for (src, dest) in bill_related:
        if src not in seen_nodes:
            vertices.append(src)
            seen_nodes.add(src)
        if dest not in seen_nodes:
            vertices.append(dest)
            seen_nodes.add(dest)

        edges.append(src+", "+dest+", "+str(bill_related[(src,dest)]))

    with open("systemg/related_bills_vertices.txt", "w") as f:
        vertices = "\n".join(vertices)
        f.write(vertices)
    with open("systemg/related_bills_edges.txt", "w") as f:
        edges = "\n".join(edges)
        f.write(edges)

    vertices = ["Node_Name"] + _IMPACT_WORDS
    edges = ["Node", "Impact", "Count"]
    # count_max = 400
    for idx, bill in enumerate(bill_impact):
        if idx == 300:
            break
        vertices.append(bill)
        impact_vector = bill_impact[bill]

        for i in range(len(impact_vector)):
            if impact_vector[i]==0 or impact_vector[i]==0.0:
                continue
            edges.append(bill+", "+_IMPACT_WORDS[i]+", "+str(impact_vector[i]))

    with open("systemg/impact_vertices.txt","w") as f:
        vertices = "\n".join(vertices)
        f.write(vertices)
    with open("systemg/impact_edges.txt","w") as f:
        edges = "\n".join(edges)
        f.write(edges)



if __name__ == '__main__':
    visualize()
