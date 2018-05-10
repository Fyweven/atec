# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from utils.config import opt

class DSSM(nn.Module):
    def __init__(self):
        super(DSSM, self).__init__()
        self.model_name = 'DSSM'
        # self.encoder = nn.Embedding(opt.char_size, opt.embedding_dim)
        self.l1 = nn.Sequential(
            nn.Linear(opt.char_size, opt.L1_N),
            nn.BatchNorm1d(opt.L1_N),
            nn.ReLU(inplace=True)
        )

        self.l2 = nn.Sequential(
            nn.Linear(opt.L1_N, opt.L2_N_N),
            nn.BatchNorm1d(opt.L2_N),
            nn.ReLU(inplace=True)
        )

        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, ques1, ques2):
        ques1_l1 = self.l1(ques1)
        ques1_l2 = self.l2(ques1_l1)

        ques2_l1 = self.l1(ques2)
        ques2_l2 = self.l2(ques2_l1)

        sim = self.cos(ques1_l2, ques2_l2)
        out = F.relu(sim)
        return out
