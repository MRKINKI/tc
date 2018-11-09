# -*- coding: utf-8 -*-
import jieba


def seg_words(samples):
    for sample in samples:
        segs = jieba.lcut(sample['content'])
        sample['seg_content'] = segs
