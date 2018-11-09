# -*- coding: utf-8 -*-

import logging
from main.utils.vocab import DataVocabs
import os
import pickle
from data import Dataset
from config import Config
from main.dl.dlmodel import DLModel


def prepare(args):
    logger = logging.getLogger("rc")
    logger.info('train test split...')
#    train_test_split(args.all_file, args.train_file, args.test_file, args.train_rate)
    sen_data = Dataset(args)
    sen_data.build_vocab()
    
    with open(args.data_path, 'wb') as fout:
        pickle.dump(sen_data, fout)
    logger.info('Done with preparing!')


def train(args):
    
    with open(args.data_path, 'wb') as fin:
        dataset = pickle.load(fin)
        
    model = DLModel(args, dataset.src_vocab.embeddings)
    
    if args.cuda:
        model.cuda()
        
    for tgt_field in dataset.tgt_info:
        for batch, pad_sentence_size in dataset.gen_mini_batches('train', args.batch_size, tgt_field):
            model.update(batch, pad_sentence_size)


if __name__ == '__main__':
    prepare(Config)
