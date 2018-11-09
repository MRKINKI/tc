import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, opts, pad_sentence_size, label_num):
        super(TextCNN, self).__init__()
        self.opts = opts
        self.label_num = label_num
        self.pad_sentence_size = pad_sentence_size
        # self.out_channel = config.out_channel
        
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

        self.lin = nn.Linear(3*self.out_channels, self.label_num)

    @staticmethod
    def get_feature_map_size(kernel_size, sentence_size, stride=1, padding=0):
        return (sentence_size + 2*padding - kernel_size) / stride + 1

    def forward(self, batch):

        x1 = self.conv3_Modules(batch).view(-1, self.out_channels)
        x2 = self.conv4_Modules(batch).view(-1, self.out_channels)
        x3 = self.conv5_Modules(batch).view(-1, self.out_channels)

        x = torch.cat((x1, x2, x3), 1)


        x = self.lin(x)
        x = x.view(-1, self.label_num)

        return x