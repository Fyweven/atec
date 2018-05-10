# -*- coding: utf-8 -*-

class Config(object):
    coding = 'UTF-8-SIG'

    lr = 1e-3
    max_len = 32
    batch_size = 64
    char_size = 1718
    embedding_dim = 128
    L1_N = 400
    L2_N = 120

    orig_data_path = '/data/atec_nlp_sim_train.csv'
    char_idx_path = '/data/pkl/char2idx.pkl'
    plot_root_path = '/data/'

opt = Config()