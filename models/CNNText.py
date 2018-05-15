# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from utils.config import opt

class CNNText(nn.Module):
    def __init__(self):
        super(CNNText, self).__init__()
        self.model_name = 'CNNText'
        if opt.baseWord:
            self.encoder = nn.Embedding(opt.word_size, opt.embedding_dim)
        else:
            self.encoder = nn.Embedding(opt.char_size, opt.embedding_dim)
        # if opt.word2vec:
        #     self.encoder.weight.data.copy_(torch.from_numpy(np.load(opt.word_embed_path)))

        self.text_convs = nn.ModuleList([ nn.Sequential(
                            nn.Conv1d(in_channels=opt.embedding_dim,
                                      out_channels=opt.text_dim,
                                      kernel_size=kernel_size),
                            nn.BatchNorm1d(opt.text_dim),
                            nn.ReLU(inplace=True),
                            nn.MaxPool1d(kernel_size=(opt.max_len - kernel_size +1))
                    )
        for kernel_size in opt.kernel_sizes])

        self.dropout = nn.Dropout(opt.dropout)
        self.fc = nn.Sequential(
            nn.Linear(len(opt.kernel_sizes)*(2*opt.text_dim), opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size, opt.num_classes)
        )

    def forward(self, text1, text2):
        text1 = self.encoder(text1)
        text2 = self.encoder(text2)
        # if opt.static:
        #     text1.detach()
        #     text2.detach()
        text1_out = [text_conv(text1.permute(0, 2, 1)) for text_conv in self.text_convs]
        text2_out = [text_conv(text1.permute(0, 2, 1)) for text_conv in self.text_convs]
        conv_out = torch.cat((text1_out+text2_out), dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        reshaped = self.dropout(reshaped)
        logits = self.fc(reshaped)
        return logits


if __name__ == '__main__':
    opt.max_len = 50
    m = CNNText()
    print(m)
    print(m.model_name)
    text = torch.autograd.Variable(torch.arange(0, 10*opt.max_len).view(10, opt.max_len)).long()
    print  (m(text, text))


