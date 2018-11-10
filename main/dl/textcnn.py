import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, opts, label_num, embedding):
        super(TextCNN, self).__init__()
        self.opts = opts
        self.label_num = label_num
        self.pad_sentence_size = opts.max_sentence_size
        
        embedding = torch.tensor(embedding).float()
        self.embedding = torch.nn.Embedding(embedding.size(0),
                                            embedding.size(1),
                                            padding_idx=0)
        self.embedding.weight.data = embedding
        
        
        self.conv3_Modules = nn.Sequential(
                                nn.Conv2d(in_channels=1,
                                          out_channels=self.opts.out_channels,
                                          kernel_size=(3, self.opts.embedding_size)),
                                nn.ReLU(),
                                nn.MaxPool2d((self.get_feature_map_size(3, self.pad_sentence_size), 1)),
        )
        self.conv4_Modules = nn.Sequential(
                                nn.Conv2d(in_channels=1,
                                          out_channels=self.opts.out_channels,
                                          kernel_size=(4, self.opts.embedding_size)),
                                nn.ReLU(),
                                nn.MaxPool2d((self.get_feature_map_size(4, self.pad_sentence_size), 1)),
        )
        
        self.conv5_Modules = nn.Sequential(
                                nn.Conv2d(in_channels=1,
                                          out_channels=self.opts.out_channels,
                                          kernel_size=(5, self.opts.embedding_size)),
                                nn.ReLU(),
                                nn.MaxPool2d((self.get_feature_map_size(5, self.pad_sentence_size), 1)),
        )

        self.lin = nn.Linear(3*self.opts.out_channels, self.label_num)
        self.criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def get_feature_map_size(kernel_size, sentence_size, stride=1, padding=0):
        feature_map_size = (sentence_size + 2*padding - kernel_size) / stride + 1
        return int(feature_map_size)

    def forward(self, batch, label):
        
        batch = self.embedding(batch)
        batch = batch.view(batch.size(0), 1, self.pad_sentence_size, self.opts.embedding_size)
        # batch = batch.float()
        # print(batch)
        x1 = self.conv3_Modules(batch).view(-1, self.opts.out_channels)

        x2 = self.conv4_Modules(batch).view(-1, self.opts.out_channels)
        x3 = self.conv5_Modules(batch).view(-1, self.opts.out_channels)

        x = torch.cat((x1, x2, x3), 1)

        x = self.lin(x)
        x = x.view(-1, self.label_num)
        loss = self.criterion(x, label)
        return loss