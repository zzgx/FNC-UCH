import pdb

import torch
from torch import nn
from torch.nn import functional as F


class TextNet(nn.Module):
    def __init__(self, y_dim, bit, norm=True, mid_num1=1024*8, mid_num2=1024*8, hiden_layer=2):

        super(TextNet, self).__init__()
        self.module_name = "txt_model"
        self.hiden_layer = hiden_layer
        mid_num1 = mid_num1 if self.hiden_layer > 1 else bit
        modules = [nn.Linear(y_dim, mid_num1)]
        if self.hiden_layer >= 2:
            modules += [nn.ReLU(inplace=True)]
            pre_num = mid_num1
            for i in range(self.hiden_layer - 2):
                if i == 0:
                    modules += [nn.Linear(mid_num1, mid_num2), nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Linear(mid_num2, mid_num2), nn.ReLU(inplace=True)]
                pre_num = mid_num2
            modules += [nn.Linear(pre_num, bit)]
        self.fc = nn.Sequential(*modules)
        #self.apply(weights_init)
        self.norm = norm

    def forward(self, x):
        # x.dtype = torch.float16 AutoModel
        # x.dtype = torch.float32 FlagModel
        out = self.fc(x).tanh()
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            out = out / norm_x
        return out
