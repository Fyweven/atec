# -*- coding: utf-8 -*-
import pickle
from config import opt
from tqdm import tqdm
import os
from io import open
path=os.path.abspath('..')

with open(path+opt.word_idx_path, 'rb') as f:
    w2i = pickle.load(f)

def wordToIdx(word):
    if word in w2i:
        return w2i[word]
    else:
        return len(w2i)-1

def textToIdx(text):
    words = text.strip().split()
    if len(words) > opt.max_len:
        words = words[:opt.max_len]
    idxs = [wordToIdx(w) for w in words]
    for i in range(len(idxs), opt.max_len):
        idxs.append(0)
    return idxs


def getWordData(seg_path):
    ids = []
    ques1_feat = []
    ques2_feat = []
    labels = []
    with open(path+seg_path, 'r', encoding=opt.coding) as f:
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
    # for w,i in w2i.items():
    #     print w
    #     print i
    text = u'怎么 更改 花呗 手机号码	我 的 花呗 是 以前 的 手机号码 ，'
    print textToIdx(text)
    # datas = getWordData(opt.seg_data_path)
