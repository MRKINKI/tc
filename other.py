import pandas as pd


#path = './data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'
#path = './data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
#df = pd.read_csv(path, encoding='utf-8')

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


embedding = nn.Embedding(10, 3)

input = torch.LongTensor([[1,2,4,5], [6,7,8,9]])

dd = embedding(input)

# print(dd)
print(dd.size(0))
dd = dd.view(dd.size(0), 1, 4, 3)

print(dd)
print(dd.size())
# print(dd)

# m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding

# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(1, 2, (2, 3))
# input = torch.randn(1, 1, 50, 100)
output = m(dd).detach()

print(output)
print(output.size())

c = F.relu(output)
print(c)

mm = nn.MaxPool2d((3, 1))
bb = mm(c)

print(bb)
# print(output)

# x = torch.randn(4, 4)
# x = x.view(2, -1)
# print(x)