import torch
import copy


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='lienar'):
        '''
        Linear layer with normal initialization
        to avoid large or small weights
        -> in_dim: dimension of input
        -> out_dim: dimension of output
        -> bias: boolean wheither to add bias term to weights
        -> w_init: xavier initialization weight gain
        '''
        super(LinearNorm, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.liner.weight,
            gain=torch.nn.init.calculate_gain(w_init))
    
    def forward(self, x):
        return self.linear(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1,
        padding=0, dilation=1, bias=True, w_init='lienar'):
        '''
        1D Convolution layer with normal initialization
        to avoid large or small weights
        -> in_ch: channels of input
        -> in_dim: channels of output
        -> kernel_size: convolution window size
        -> stride: step size
        -> padding: size of the padding for input
        -> dilation: dilation rate
        -> bias: boolean wheither to add bias term to weights
        -> w_init: xavier initialization weight gain
        ''' 
        self.conv = torch.nn.Conv1d(in_ch, out_ch, kernel_size,
            stride, padding, dilation, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init))    

    def forward(self, x):
        return self.conv(x)    


def clone_layers(layer, copy_num):
    """ Quickly Create the Repetitive Modules"""
    layers = torch.nn.ModuleList([copy.deepcopy(layer) for i in range(copy_num)])
    return layers