# -*- coding: utf-8 -*-
import torch


class Config(object):
    coding = 'UTF-8-SIG'

    lr = 1e-3
    lr2 = 5e-4  # embedding层的学习率
    max_len = 32
    word_len = 16
    batch_size = 64
    char_size = 1718
    word_size = 3346
    embedding_dim = 128
    L1_N = 400
    L2_N = 120

    text_dim = 100
    kernel_sizes = [1, 2, 3, 4]
    dropout = 0.5
    linear_hidden_size = 128
    num_classes = 2
    num_layers = 1  # LSTM layers
    hidden_size = 128  # LSTM hidden size
    kmax_pooling = 2  # CNN2:3 best:2


    max_epoch = 5
    use_gpu = torch.cuda.is_available()
    # use_gpu = False

    baseWord = True
    word2vec = True

    orig_data_path = '/data/atec_nlp_sim_train.csv'
    seg_data_path = '/data/atec_nlp_sim_train_seg.csv'
    char_idx_path = '/data/pkl/char2idx.pkl'
    word_idx_path = '/data/pkl/word2idx.pkl'
    pkl_root_path = '/data/pkl/'
    word_embed_path = '/data/pkl/word_embed.npy'
    plot_root_path = '/data/'
    model_root_path = '/data/model/'
    word_dict_path = '/data/word_dict.txt'

opt = Config()