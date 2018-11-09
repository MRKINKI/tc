#!/user/bin/env python
# -*- coding:utf-8 -*-

import os


class Config:
    model_save_path = './data/'
    train_data_path = './data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
    validate_data_path = './data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'
    test_data_path = "./data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv"
    test_data_predict_out_path = "./data/test_result.csv"
    data_path = './data/sen.dat'
    
    cuda = True
    min_count = 2
    embedding_size = 128
    out_channels = 128
    model = 'textcnn'
    max_sentence_size = 500
    batch_size = 32
