# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader
from scripts import dataLoad
from utils.config import opt
import time
from tqdm import tqdm
from torch.autograd import Variable
from models.CNNText import CNNText
from models.LSTMText import LSTMText
import os
path=os.path.abspath('..')

def train(model):
    print('加载数据。。。')
    dataset = dataLoad.MyDataset(train=True)
    print ('数据加载完成！')

    weight = torch.Tensor([0.3,0.7])
    if opt.use_gpu:
        weight = weight.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(1, opt.max_epoch+1):
        print ('')
        print('*' * 6 + '[epoch {}/{}]'.format(epoch,opt.max_epoch) +'*' * 6)
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        dataset.setTrain(True)
        train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=False)
        step = 0
        time.sleep(0.1)
        for data in tqdm(train_loader):
            txt1, txt2, label = data
            txt1, txt2, label = Variable(txt1), Variable(txt2), Variable(label)
            if opt.use_gpu:
                txt1, txt2, label = txt1.cuda(), txt2.cuda(), label.cuda()
            optimizer.zero_grad()
            out = model(txt1, txt2)
            loss = criterion(out,label)

            running_loss += loss.data.mean() * label.size(0)
            running_acc += get_acc(out,label)

            loss.backward()
            optimizer.step()

            step+=1
            if step % 100 == 0:
                print('Loss: {:.6f}, Acc: {:.6f}'.format(running_loss / (opt.batch_size * step), running_acc / (opt.batch_size * step)))
        print ('')
        base = 'word' if opt.baseWord else 'char'
        torch.save(model.state_dict(), path + opt.model_root_path + '%s_%s_%s.pkl' % (model.model_name,base,str(epoch)))
        eval(model, dataset)
    print ('*' * 6 + 'END' +'*' * 6)

def get_acc(out, label):
    # pred = torch.round(F.sigmoid(out))
    pred = torch.argmax(out, 1)
    num_correct = (pred == label).sum()
    return num_correct.item()

def eval(model, dataset):
    model.eval()
    dataset.setTrain(False)
    test_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    tp, tn, fn, fp = 0, 0, 0, 0
    print('*' * 3 + 'eval' + '*' * 3)
    time.sleep(0.1)
    for data in tqdm(test_loader):
        txt1, txt2, label = data
        txt1, txt2, label = Variable(txt1), Variable(txt2), Variable(label)
        if opt.use_gpu:
            txt1, txt2, label = txt1.cuda(), txt2.cuda(), label.cuda()
        out = model(txt1, txt2)
        pred = torch.argmax(out, 1)
        tp += ((pred == 1) & (label == 1)).cpu().sum().item()
        tn += ((pred == 0) & (label == 0)).cpu().sum().item()
        fn += ((pred == 0) & (label == 1)).cpu().sum().item()
        fp += ((pred == 1) & (label == 0)).cpu().sum().item()
    print (tp, tn, fn, fp)
    p,r,a,f1 = get_all_f1(tp, tn, fn, fp)
    print (p,r,a,f1)
    return p,r,a,f1

def get_all_f1(tp, tn, fn, fp):
    p = 1.0*tp/(tp+fp) if (tp+fp)!=0 else 0.0
    r = 1.0*tp/(tp+fn) if (tp+fn)!=0 else 0.0
    a = 1.0*(tp+tn)/(tp+fp+tn+fn)
    f1 = 2.0*p*r/(p+r) if(p+r)!=0 else 0.0
    return p, r, a, f1


def initNetParams(net):
    '''''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            init.xavier_uniform(m.weight)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            init.constant(m.bias, 0)

if __name__ == '__main__':
    if opt.use_gpu:
        torch.cuda.set_device(2)
    model = LSTMText()
    print(model.model_name)
    print 'use_gpu' if opt.use_gpu else 'use_cpu'
    print ('')
    if opt.use_gpu:
        model.cuda()
    initNetParams(model)
    # print(model)
    train(model)