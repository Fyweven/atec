# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
from torch.nn import init
from torch.utils.data import DataLoader
from scripts import dataLoad
from utils.config import opt
import time
from tqdm import tqdm
from torch.autograd import Variable
from models import dssm

def train(model):
    print('加载数据。。。')
    dataset = dataLoad.MyDataset(train=True)
    print ('数据加载完成！')

    criterion = nn.BCELoss()

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
            txt, label = data
            txt = Variable(txt)
            label = Variable(label)
            if opt.use_gpu:
                txt, label = txt.cuda(), label.cuda()
            optimizer.zero_grad()
            out = model(txt)
            loss = criterion(out,label)
            running_loss += loss.data.mean() * label.size(0)
            running_acc += get_acc(out,label)

            loss.backward()
            optimizer.step()

            step+=1
            if step % 100 == 0:
                print('Loss: {:.6f}, Acc: {:.6f}'.format(running_loss / (opt.batch_size * step), running_acc / (opt.batch_size * step)))
        print ('')
        # base = 'word' if opt.baseWord else 'char'
        base = 'oneHot'
        torch.save(model.state_dict(), opt.model_root_path + '%s_%s_%s.pkl' % (model.model_name,base,str(epoch)))

def eval(model, dataset):
    model.eval()
    dataset.setTrain(False)
    test_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    eval_acc = 0.0
    test_num = 0
    res = []
    print('*' * 3 + 'eval' + '*' * 3)
    time.sleep(0.1)
    for data in tqdm(test_loader):
        txt, label = data
        txt = Variable(txt)
        label = Variable(label)
        if opt.use_gpu:
            txt = txt.cuda()
            label = label.cuda()
        out = model(txt)
        _, pred = torch.max(out, 1)
        res += pred.data.tolist()
        eval_acc += get_acc(out, label)
        test_num += label.size(0)


def get_acc(out, label):
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    return num_correct.data[0]

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
        torch.cuda.set_device(3)
    model = dssm.DSSM()
    print(model.model_name)
    print('word' if opt.baseWord else 'char')
    if opt.use_gpu:
        model.cuda()
    # initNetParams(model)
    # print(model)
    train(model)