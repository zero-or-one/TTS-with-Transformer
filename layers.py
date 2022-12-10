import torch


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        '''
        Linear layer with normal initialization
        to avoid large or small weights
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean wheither to add bias term to weights
        :param w_init: xavier initialization weight gain
        '''
        super(LinearNorm, self).__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.fc.weight,
            gain=torch.nn.init.calculate_gain(w_init))
    
    def forward(self, x):
        return self.fc(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_dim=1, stride=1,
        padding=0, dilation=1, bias=True, w_init='linear'):
        '''
        1D Convolution layer with normal initialization
        to avoid large or small weights
        :param in_ch: channels of input
        :param in_dim: channels of output
        :param kernel_size: convolution window size
        :param stride: step size
        :param padding: size of the padding for input
        :param dilation: dilation rate
        :param bias: boolean wheither to add bias term to weights
        :param w_init: xavier initialization weight gain
        ''' 
        super(ConvNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_ch, out_ch, kernel_dim,
            stride, padding, dilation, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init))    

    def forward(self, x):
        return self.conv(x)    