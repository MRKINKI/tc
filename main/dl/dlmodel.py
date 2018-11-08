# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn.functional as F
from .textcnn import TextCNN


class dlmodel:
    def __init__(self, opts, embedding=None, padding_idx=0):
        
        self.embedding = torch.nn.Embedding(embedding.size(0),
                                            embedding.size(1),
                                            padding_idx=padding_idx)
        if opts.model == 'textcnn':
            self.network = TextCNN()
        
    
    def update(self, batch):
        pass
    
    def cuda(self):
        self.network.cuda()
