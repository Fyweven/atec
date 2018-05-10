# -*- coding: utf-8 -*-
import pickle
from utils.config import opt
from tqdm import tqdm
import os
from io import open
import matplotlib.pyplot as plt
import chardet
path=os.path.abspath('..')

def getChar2Idx(file_path, idx_path):
    all_char = set()
    with open(path+file_path, 'r', encoding=opt.coding) as f:
        for line in tqdm(f.readlines()):
            items = line.strip().split('\t')
            assert len(items) == 4
            text = items[1].strip()+items[2].strip()
            for c in text:
                all_char.add(c)
    char2idx = {}
    for i, c in enumerate(list(all_char)):
        char2idx[c] = i + 1
        char2idx[u'xxPaddingxx'] = 0
    with open(path+idx_path, 'wb') as f:
        pickle.dump(char2idx, f)

def getCharLen(file_path):
    char_len = []
    with open(path+file_path, 'r', encoding=opt.coding) as f:
        for line in f.readlines():
            items = line.strip().split('\t')
            assert len(items) == 4
            text1 = items[1].strip()
            char_len.append(len(text1))
            text2 = items[2].strip()
            char_len.append(len(text2))
    return char_len

def plotHist(len_array, plot_name):
    plt.figure()
    plt.title(plot_name)
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.hist(len_array, bins=50, normed=1, alpha=.5)
    plt.savefig(path+opt.plot_root_path+plot_name)

def getDataCoding(file_path):
    f = open(path + file_path, 'rb')
    data = f.read()
    print chardet.detect(data)


if __name__ == '__main__':
    getDataCoding(opt.orig_data_path)
    # getChar2Idx(opt.orig_data_path,opt.char_idx_path)
    # char_len = getCharLen(opt.orig_data_path)
    # count = 0
    # for l in char_len:
    #     if l>opt.max_len:
    #         count+=1
    # print count
    # print count*1.0/len(char_len)
    # plotHist(char_len, 'ques_char_len')