# -*- coding: utf-8 -*-
import torch


class Config(object):
    coding = 'UTF-8-SIG'

    lr = 1e-3
    max_len = 32
    batch_size = 64
    char_size = 1718
    embedding_dim = 128
    L1_N = 400
    L2_N = 120

    text_dim = 100
    kernel_sizes = [1, 2, 3, 4]
    dropout = 0.5
    linear_hidden_size = 128
    num_classes = 2


    max_epoch = 1
    use_gpu = torch.cuda.is_available()

    baseWord = False

    orig_data_path = '/data/atec_nlp_sim_train.csv'
    char_idx_path = '/data/pkl/char2idx.pkl'
    plot_root_path = '/data/'
    model_root_path = '/data/model/'

opt = Config()