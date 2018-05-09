# -*- coding: utf-8 -*-
import pickle
from config import opt
from tqdm import tqdm
import os
from io import open
import csv
path=os.path.abspath('..')

with open(path+opt.char_idx_path, 'rb') as f:
    char2idx = pickle.load(f)

def charToIdx(char):
    if char in char2idx:
        return char2idx[char]
    else:
        return len(char2idx)

def textToIdx(text):
    if len(text)>opt.max_len:
        text = text[:opt.max_len]
    idx = [charToIdx(c) for c in text]
    for i in range(len(idx),opt.max_len):
        idx.append(0)
    return idx

def getCharData(file_path):
    ids = []
    ques1_feat = []
    ques2_feat = []
    labels = []
    with open(path+file_path, 'r', encoding=opt.coding) as f:
        for line in tqdm(f.readlines()):
            items = line.strip().split('\t')
            assert len(items) == 4
            ids.append(int(items[0].strip()))
            labels.append(int(items[3].strip()))
            text1 = items[1].strip()
            text2 = items[2].strip()
            ques1_feat.append(textToIdx(text1))
            ques2_feat.append(textToIdx(text2))
    return ids, (ques1_feat, ques2_feat, labels)




if __name__ == '__main__':
    # datas = getCharData(opt.orig_data_path)
    print len(char2idx)