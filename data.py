# -*- coding: utf-8 -*-
import pandas as pd
# import logging
from main.utils.tokenize import seg_words
from main.utils.vocab import Vocab


class Dataset(object):

    def __init__(self, opts):
        # self.logger = logging.getLogger("rc")
        # self.feature_extract = FeatureExtract()
        self.min_count = opts.min_count
        self.embedding_size = opts.embedding_size
        self.train_set, self.dev_set, self.test_set = [], [], []

        self.train_set, self.tgt_info = self._load_dataset(opts.train_data_path, train=True)

        self.dev_set, _ = self._load_dataset(opts.validate_data_path)

        self.test_set, _ = self._load_dataset(opts.test_data_path)
        self.src_vocab = None
        self.tgt_vocab = None
        # self.logger.info('train_set size: {}'.format(len(self.train_set)))
        # self.logger.info('dev_set size: {}'.format(len(self.dev_set)))

    def _load_dataset(self, data_path, header=0, encoding="utf-8", train=False):
        data_df = pd.read_csv(data_path, header=header, encoding=encoding)
        content = list(data_df['content'])
        # print(data_df.columns[:])
        if train:
            tgt_info = data_df.iloc[:, 2:]
        else:
            tgt_info = ''
        return seg_words(content), tgt_info

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
        for column in self.tgt_info.columns:
            tgt_vocab = Vocab(initial_tokens=False, lower=False)
            for tgt in self.tgt_info[column]:
                tgt_vocab.add(tgt)
            tgt_vocab_dict[column] = tgt_vocab
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab_dict

    def _one_mini_batch(self, data, indices):

        raw_data = [data[i] for i in indices]
        batches = []
        for sidx, sample in enumerate(raw_data):
            batch_data = {}
            batch_data['question_word_ids'] = sample['question_word_ids']
            batch_data['context_word_ids'] = sample['context_word_ids']
            batch_data['context_word'] = sample['context_word']
            batch_data['context_ner_ids'] = sample['context_ner_ids']
            batch_data['context_pos_ids'] = sample['context_pos_ids']
            batch_data['context_feature'] = sample['context_feature'] 
            if 'answer_spans' in sample:
                batch_data['start_id'] = sample['answer_start']
                batch_data['end_id'] = sample['answer_end']
                batch_data['answer'] = sample['answer']
            else:
                batch_data['start_id'] = 0
                batch_data['end_id'] = 0
                batch_data['answer'] = ''
            batches.append(batch_data)
        return batches

    def _dynamic_padding(self, batch_data, pad_id):

        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len

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
                for token in sample:
                    yield token
                    

    def convert_to_ids(self, data_vocabs):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_word_ids'] = data_vocabs.word_vocab.convert_to_ids(sample['question_word'])
                sample['context_word_ids'] = data_vocabs.word_vocab.convert_to_ids(sample['context_word'])
                sample['context_ner_ids'] = data_vocabs.ner_vocab.convert_to_ids(sample['context_ner'])
                sample['context_pos_ids'] = data_vocabs.pos_vocab.convert_to_ids(sample['context_pos'])
#                for passage in sample['passages']:
#                    passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])

    def gen_mini_batches(self, set_name, batch_size, shuffle=True):

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
            yield self._one_mini_batch(data, batch_indices)