# -*- coding: utf-8 -*-
import jieba


def seg_words(contents):
    contents_segs = list()
    for content in contents:
        segs = jieba.lcut(content)
        contents_segs.append(list(segs))
    return contents_segs