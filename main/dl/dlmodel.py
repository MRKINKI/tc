# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn.functional as F
from .textcnn import TextCNN


class DLModel:
    def __init__(self, opts, tgt_num, embedding=None, padding_idx=0):
        

        self.opts = opts
        if opts.model == 'textcnn':
            self.network = TextCNN(opts, tgt_num, embedding)
            
        # parameters = [p for p in self.network.parameters() if p.requires_grad]
        # self.optimizer = optim.Adamax(parameters,
        #                        weight_decay=opt['weight_decay'])
        self.optimizer = torch.optim.Adamax(self.network.parameters())
    
    # 词向量
    def update(self, batch):
        self.network.train()
        src = [t['sentence_word_ids'] for t in batch]
        # print([len(t) for t in src])
        tgt = [t['tgt'] for t in batch]
        
        src = torch.LongTensor(src)
        tgt = torch.LongTensor(tgt)
        
        if self.opts.cuda:
            src = src.cuda()
            tgt = tgt.cuda()
        self.optimizer.zero_grad()
        loss = self.network(src, tgt)
        loss.backward()
        self.optimizer.step()
        
        
    def cuda(self):
        self.network.cuda()
        