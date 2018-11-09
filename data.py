# -*- coding: utf-8 -*-
import pandas as pd
# import logging
from main.utils.tokenize import seg_words
from main.utils.vocab import Vocab
import numpy as np


class Dataset(object):

    def __init__(self, opts):
        # self.logger = logging.getLogger("rc")
        # self.feature_extract = FeatureExtract()
        self.max_sentence_size = opts.max_sentence_size
        self.min_count = opts.min_count
        self.embedding_size = opts.embedding_size
        self.train_set, self.dev_set, self.test_set = [], [], []
        self.un_tgt_field = ['id', 'content', 'seg_content']

        self.train_set = self._load_dataset(opts.train_data_path, train=True)

        self.dev_set = self._load_dataset(opts.validate_data_path)

        self.test_set = self._load_dataset(opts.test_data_path)
        self.tgt_info = self._get_tgt_info()
        self.src_vocab = None
        self.tgt_vocab = None
        # self.logger.info('train_set size: {}'.format(len(self.train_set)))
        # self.logger.info('dev_set size: {}'.format(len(self.dev_set)))

    def _load_dataset(self, data_path, header=0, encoding="utf-8", train=False):
        data = pd.read_csv(data_path, header=header, encoding=encoding).to_dict('records')
        seg_words(data)
        return data
        
    def _get_tgt_info(self):
        return list(filter(lambda x:True if x not in self.un_tgt_field else False, 
                    list(self.train_set[0].keys())))

    def build_vocab(self):
        src_vocab = Vocab()
        for word in self.word_iter('train'):
            src_vocab.add(word)
        # unfiltered_vocab_size = src_vocab.size()
        src_vocab.filter_tokens_by_cnt(min_cnt=self.min_count)
        # filtered_num = unfiltered_vocab_size - src_vocab.size()
        # self.logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
        #                  src_vocab.size()))
        
        # self.logger.info('Assigning embeddings...')
        src_vocab.randomly_init_embeddings(self.embedding_size)
        
        tgt_vocab_dict = {}
        for tgt_field in self.tgt_info:
            tgt_vocab = Vocab(initial_tokens=False, lower=False)
            for sample in self.train_set:
                tgt = sample[tgt_field]
                tgt_vocab.add(tgt)
            tgt_vocab_dict[tgt_field] = tgt_vocab
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab_dict
        self.convert_to_ids(self.src_vocab, self.tgt_vocab)

    def _one_mini_batch(self, data, indices, tgt_field, set_name):

        raw_data = [data[i] for i in indices]
        batch = []
        for sidx, sample in enumerate(raw_data):
            batch_data = {}
            batch_data['sentence_word_ids'] = sample['sentence_word_ids']
            if set_name in ['train', 'dev']:
                batch_data['tgt'] = sample[tgt_field]

            batch.append(batch_data)
        batch, pad_sentence_size = self._dynamic_padding(batch, 0)
        return batch, pad_sentence_size

    def _dynamic_padding(self, batch_data, pad_id):

        pad_sentence_size = min(self.max_sentence_size, 
                                max([len(t['sentence_word_ids']) for t in batch_data]))
        
        for sub_batch_data in batch_data:
            ids = sub_batch_data['sentence_word_ids']
            sub_batch_data['sentence_word_ids'] = ids + [pad_id] * (pad_sentence_size - len(ids))[:pad_sentence_size]

        return batch_data, pad_sentence_size

    def word_iter(self, set_name=None):

        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['seg_content']:
                    yield token
                    

    def convert_to_ids(self, src_vocab, tgt_vocab):

        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if not len(data_set):
                continue
            for sample in data_set:
                sample['sentence_word_ids'] = src_vocab.word_vocab.convert_to_ids(sample['seg_content'])
                for tgt_field in self.tgt_info:
                    if tgt_field in sample:
                        sample[tgt_field + '_id'] = tgt_vocab[tgt_field].get_id(sample[tgt_field])

    def gen_mini_batches(self, set_name, batch_size, tgt_field, shuffle=True):

        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, tgt_field, set_name)
