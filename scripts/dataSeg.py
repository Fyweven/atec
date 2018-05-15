# -*- coding: utf-8 -*-
from io import open
import os
from tqdm import tqdm
from utils.config import opt
import jieba
import re
from gensim.models import Word2Vec
import numpy as np
import pickle

path=os.path.abspath('..')

jieba.load_userdict(path+opt.word_dict_path)
jieba_seg = jieba.dt

# 同义词表
pattern_huabei = re.compile(u'蚂蚁花唄|花唄|蚂蚁花呗|蚂蚁花被|蚂蚁花贝|蚂蚁花吧|花被|花贝|花吧')
pattern_jiebei = re.compile(u'蚂蚁借唄|借唄|蚂蚁借呗|借被|借贝|借吧|蚂蚁借被|蚂蚁借贝|蚂蚁借吧')
pattern_char = re.compile(u'\*\*\*')
pattern_zhima = re.compile(u'芝麻信用')

def textSeg(text):
    text = re.sub(pattern_char, u'*', text)
    text = re.sub(pattern_huabei, u'花呗', text)
    text = re.sub(pattern_jiebei, u'借呗', text)
    text = re.sub(pattern_zhima, u'芝麻', text)
    words = jieba_seg.cut(text)
    text = ' '.join(words)
    return text

def segData(file_path, seg_path):
    res = []
    with open(path+file_path, 'r', encoding=opt.coding) as f:
        for line in tqdm(f.readlines()):
            items = line.strip().split('\t')
            assert len(items) == 4
            items[1] = textSeg(items[1].strip())
            items[2] = textSeg(items[2].strip())
            res.append('\t'.join(items))
    with open(path+seg_path, 'w', encoding=opt.coding) as f:
        f.write('\n'.join(res))

def getSentences(seg_path):
    sentences = []
    with open(path+seg_path, 'r', encoding=opt.coding) as f:
        for line in f.readlines():
            items = line.strip().split('\t')
            assert len(items) == 4
            sentences.append(items[1].strip().split())
            sentences.append(items[2].strip().split())
    return sentences

def word2vec():
    sentences = getSentences(opt.seg_data_path)
    print(len(sentences))
    model = Word2Vec(sentences, size=opt.embedding_dim, window=5, min_count=3, workers=4)
    model.save(path+opt.pkl_root_path + 'word_vec.model')

def saveWordEmbed():
    mod = Word2Vec.load(path+opt.pkl_root_path + 'word_vec.model')
    word_vectors = mod.wv
    all_words = word_vectors.index2word
    word2idx = {u'xxPaddingxx': 0}
    word_embed = np.zeros((len(all_words) + 2, opt.embedding_dim))
    print(len(all_words))
    for i, word in enumerate(all_words):
        word2idx[word] = i + 1
        word_embed[i + 1] = word_vectors[word]
    word2idx[u'xx生词xx'] = len(word2idx)
    with open(path+opt.word_idx_path, 'wb') as f:
        pickle.dump(word2idx, f)
    np.save(path+opt.word_embed_path, word_embed)


if __name__ == '__main__':
    # segData(opt.orig_data_path,opt.seg_data_path)
    word2vec()
    saveWordEmbed()