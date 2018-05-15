# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from utils.config import opt
import numpy as np
import os
path=os.path.abspath('..')


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class LSTMText(nn.Module):
    def __init__(self):
        super(LSTMText, self).__init__()
        self.model_name = 'LSTMText'

        if opt.baseWord:
            self.encoder = nn.Embedding(opt.word_size, opt.embedding_dim)
        else:
            self.encoder = nn.Embedding(opt.char_size, opt.embedding_dim)

        if opt.word2vec:
            self.encoder.weight.data.copy_(torch.from_numpy(np.load(path+opt.word_embed_path)))

        self.text_lstm = nn.LSTM(input_size = opt.embedding_dim,
                            hidden_size = opt.hidden_size,
                            num_layers = opt.num_layers,
                            bias = True,
                            batch_first = False,
                            bidirectional = True)
        self.dropout = nn.Dropout(opt.dropout)
        self.fc = nn.Sequential(
            nn.Linear(opt.kmax_pooling * (opt.hidden_size * 2 * 2), opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size, opt.num_classes)
        )

    def forward(self, text1, text2):
        text1, text2 = self.encoder(text1), self.encoder(text2)
        text1_out = self.text_lstm(text1.permute(1,0,2))[0].permute(1,2,0)
        text1_out = kmax_pooling((text1_out), 2, opt.kmax_pooling)
        text2_out = self.text_lstm(text2.permute(1, 0, 2))[0].permute(1, 2, 0)
        text2_out = kmax_pooling((text2_out), 2, opt.kmax_pooling)
        lstm_out = torch.cat((text1_out,text2_out), dim=1)
        reshaped = lstm_out.view(lstm_out.size(0), -1)
        reshaped = self.dropout(reshaped)
        logits = self.fc(reshaped)
        return logits


if __name__ == '__main__':
    # opt.max_len=500
    m = LSTMText()
    print(m)
    text = torch.autograd.Variable(torch.arange(0, 10*opt.max_len).view(10, opt.max_len)).long()
    print(m(text, text))