import torch
import torch.nn as nn


class RadarPCL(nn.Module):
    
    def __init__(self, convnet_params, linearnet_params):
        super(RadarPCL, self).__init__()
        
        self.convnet_params = convnet_params
        self.linearnet_params = linearnet_params

        # Check type of params
        assert isinstance(convnet_params['kernel_size'], list), 'kernel_size should be a list'
        assert len(convnet_params['kernel_size']) >= 1 and len(convnet_params['kernel_size']) != len(convnet_params['channels']), 'kernel_size should have 1 element or N elements where N is equal to the number of channels'
        if len(convnet_params['kernel_size']) == 1:
            # Partially defined
            kernel_size = convnet_params['kernel_size'] * len(convnet_params['channels'])
        else:
            # Fully defined
            kernel_size = convnet_params['kernel_size']

        assert isinstance(convnet_params['dilation'], list), 'dilation should be a list'
        assert len(convnet_params['dilation']) >= 1 and len(convnet_params['dilation']) != len(convnet_params['channels']), 'dilation should have 1 element or N elements where N is equal to the number of channels'
        if len(convnet_params['dilation']) == 1:
            # Partially defined
            dilation = convnet_params['dilation'] * len(convnet_params['channels'])
        else:
            # Fully defined
            dilation = convnet_params['dilation']

        assert isinstance(convnet_params['stride'], list), 'stride should be a list'
        assert len(convnet_params['stride']) >= 1 and len(convnet_params['stride']) != len(convnet_params['channels']), 'stride should have 1 element or N elements where N is equal to the number of channels'
        if len(convnet_params['stride']) == 1:
            # Partially defined
            stride = convnet_params['stride'] * len(convnet_params['channels'])
        else:
            # Fully defined
            stride = convnet_params['stride']

        assert isinstance(convnet_params['padding'], list), 'padding should be a list'
        assert len(convnet_params['padding']) >= 1 and len(convnet_params['padding']) != len(convnet_params['channels']), 'padding should have 1 element or N elements where N is equal to the number of channels'
        if len(convnet_params['padding']) == 1:
            # Partially defined
            padding = convnet_params['padding'] * len(convnet_params['channels'])
        else:
            # Fully defined
            padding = convnet_params['padding']

        groups = [convnet_params['groups']] * len(convnet_params['channels'])

        layers = []
        width = len(convnet_params['channels'])
        for i in range(width):
            in_channels = convnet_params['in_channels'] if i == 0 else convnet_params['channels'][i-1]
            if i == width - 1:
                layers += [self._conv(in_channels, convnet_params['channels'][i], kernel_size[i], dilation[i], stride[i], padding[i], groups[i])]
            else:
                layers += [self._conv(in_channels, convnet_params['channels'][i], kernel_size[i], dilation[i], stride[i], padding[i], groups[i])]
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

    def _conv(self, in_channels, out_channels, kernel_size, dilation, stride, padding, groups):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
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
