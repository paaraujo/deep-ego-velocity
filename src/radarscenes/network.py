import torch
import torch.nn as nn


class MyDropout(nn.Module):
    ''' Dropout a pixel through all channels.
    '''
    def __init__(self, p: float = 0.5):
        super(MyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, X):
        if self.training:
            bernoulli = torch.distributions.bernoulli.Bernoulli(probs=1-self.p)
            B, C, H, W = X.size()
            sample = bernoulli.sample((B, 1, H, W))
            sample = sample.expand(B, C, H, W).to(X.device)
            return X * sample * (1.0/(1.0-self.p))
        return X
    

class RadarPCL(nn.Module):
    
    def __init__(self, convnet_params, linearnet_params):
        super(RadarPCL, self).__init__()
        
        self.convnet_params = convnet_params
        self.linearnet_params = linearnet_params

        layers = []
        width = len(convnet_params['channels'])
        for i in range(width):
            in_channels = convnet_params['in_channels'] if i == 0 else convnet_params['channels'][i-1]
            if i == width - 1:
                layers += [self._conv(in_channels, convnet_params['channels'][i], convnet_params['kernel_size'], convnet_params['dilation'], convnet_params['stride'], convnet_params['padding'], convnet_params['groups'], 0.)]
            else:
                layers += [self._conv(in_channels, convnet_params['channels'][i], convnet_params['kernel_size'], convnet_params['dilation'], convnet_params['stride'], convnet_params['padding'], convnet_params['groups'], convnet_params['dropout'])]
        self.conv = nn.Sequential(*layers)

        layers = []
        width = len(linearnet_params['channels'])
        for i in range(width):
            output_size = torch.tensor(self._get_output_shape(self.conv, convnet_params['input_size']))
            in_channels = torch.prod(output_size, dtype=torch.int32) if i == 0 else linearnet_params['channels'][i-1]
            if i == width - 1:
                layers += [self._linear(in_channels, linearnet_params['channels'][i], True)]
            else:
                layers += [self._linear(in_channels, linearnet_params['channels'][i], False)]
        self.linear = nn.Sequential(*layers)

    def _get_output_shape(self, model, tensor_dim):
        return model(torch.rand(*(tensor_dim))).data.shape

    def _conv(self, in_channels, out_channels, kernel_size, dilation, stride, padding, groups, dropout):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            # MyDropout(dropout)  # Commented due to performance issues, applying dropout in the input array directly
        )    

    def _linear(self, in_channels, our_channels, last_activation):
        return nn.Sequential(
            nn.Linear(in_channels, our_channels, bias=False),
            nn.BatchNorm1d(our_channels),
            nn.GELU() if not last_activation else nn.Identity(),
        )    
    
    def forward(self, I):
        batch_size = I.size(0)
        out = self.conv(I)
        out = out.view(batch_size, -1)
        out = self.linear(out)
        return out
