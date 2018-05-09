import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils.config import opt

class DSSM(nn.Module):
    def __init__(self):
        super(DSSM, self).__init__()
        self.model_name = 'DSSM'
        # self.encoder = nn.Embedding(opt.char_size, opt.embedding_dim)
        self.ques_encoder = nn.Linear(opt.char_size, )