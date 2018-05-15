# -*- coding: utf-8 -*-
from utils.config import opt
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import utils.charFeature as charFeat
import utils.wordFeature as wordFeat

def getTrainData():
    if opt.baseWord:
        _, datas = wordFeat.getWordData(opt.seg_data_path)
    else:
        _, datas = charFeat.getCharData(opt.orig_data_path)
    return datas

class MyDataset(data.Dataset):
    def __init__(self, train=True, dev_ratio=.1):
        feat1, feat2, labels = getTrainData()
        self.train = train
        dev_index = -1 * int(dev_ratio * len(labels))
        train_1feats, train_2feats, train_labels = feat1[:dev_index], feat2[:dev_index], labels[:dev_index]
        self.train_1texts = torch.LongTensor(train_1feats)
        self.train_2texts = torch.LongTensor(train_2feats)
        self.train_labels = torch.LongTensor(train_labels)

        test_1feats, test_2feats, test_labels = feat1[dev_index:], feat2[dev_index:], labels[dev_index:]
        self.test_1texts = torch.LongTensor(test_1feats)
        self.test_2texts = torch.LongTensor(test_2feats)
        self.test_labels = torch.LongTensor(test_labels)

    def setTrain(self, train):
        self.train = train

    def __getitem__(self, index):
        if self.train:
            txt1, txt2, target = self.train_1texts[index], self.train_2texts[index], self.train_labels[index]
        else:
            txt1, txt2, target = self.test_1texts[index], self.test_2texts[index], self.test_labels[index]
        return txt1, txt2, target

    def __len__(self):
        if self.train:
            return len(self.train_1texts)
        else:
            return len(self.test_1texts)

if __name__ == '__main__':
    dataset = MyDataset(train=True)
    print ('数据加载完成')
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    print (len(train_loader))
    for tt in train_loader:
        txt1, txt2, l = tt
        print txt1
        print l
        break