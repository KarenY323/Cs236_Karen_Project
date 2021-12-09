import torch.nn as nn
import torch.nn.functional as F


class DCGANGenerator(nn.Module):
    def __init__(self, input_size,  image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
        super(DCGANGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.blocks = blocks
        self.height = height
        self.length = length
        self.mult = 2**blocks

        self.initial_linear = nn.Linear(input_size, hidden_size * self.mult * height//self.mult * length//self.mult)
        self.initial_activ = nn.PReLU(hidden_size * self.mult * height//self.mult * length//self.mult)
        self.initial_norm = nn.LayerNorm(hidden_size * self.mult * height//self.mult * length//self.mult)

        self.convs = nn.ModuleList(
            [ nn.Conv2d(hidden_size * 2 **(blocks - i), hidden_size * 2**(blocks - i - 1), (5, 5), padding=(2, 2)) for i in range(blocks) ]
        )
        self.activ = nn.ModuleList(
            [ nn.PReLU(hidden_size * 2**(blocks - i - 1)) for i in range(blocks) ]
        )
        self.norm = nn.ModuleList(
            [ nn.LayerNorm([hidden_size * 2 ** (blocks - i - 1), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in range(blocks) ]
        )

        self.final_conv = nn.Conv2d(hidden_size, image_channels, (5, 5), padding=(2, 2))
        self.final_activ = nn.Tanh()

    def forward(self, inputs):
        x = self.initial_linear(inputs)
        x = self.initial_activ(x)
        x = self.initial_norm(x)
        x = x.view(x.shape[0], self.hidden_size * self.mult, self.height//self.mult, self.length//self.mult)

        for i in range(self.blocks):
            x = self.convs[i](x)
            x = self.activ[i](x)
            x = self.norm[i](x)
            x = F.upsample(x, scale_factor=2)
        x = self.final_conv(x)
        x = self.final_activ(x)
        return x

class DCGANDiscriminator(nn.Module):
    def __init__(self, image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
        super(DCGANDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.blocks = blocks

        self.initial_conv = nn.Conv2d(image_channels, hidden_size, (5, 5), padding=(2, 2))
        self.initial_norm = nn.LayerNorm([hidden_size, height, length])
        self.initial_activ = nn.PReLU(hidden_size)

        self.convs = nn.ModuleList(
            [ nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** (i + 1), (5, 5), padding=(2, 2)) for i in range(blocks) ]
        )
        self.norm = nn.ModuleList(
            [ nn.LayerNorm([hidden_size * 2 ** (i + 1), height // (2 ** i), length // (2 ** i)]) for i in range(blocks) ]
        )
        self.activ = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (i + 1)) for i in range(blocks)])

        self.final_linear = nn.Linear(hidden_size * 2 ** blocks * height//(2**blocks) * length//(2**blocks), 1)

    def forward(self, inputs):
        x = self.initial_conv(inputs)
        x = self.initial_norm(x)
        x = self.initial_activ(x)

        for i in range(self.blocks):
            x = self.convs[i](x)
            x = self.norm[i](x)
            x = self.activ[i](x)
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        return x


'''
class CycleGanGenerator():
    def __init__(self, input_size,  image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
        pass

    def forward(self, inputs):
        pass

class CycleGanDiscriminator():
    def __init__(self, input_size,  image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
        pass

    def forward(self, inputs):
        pass
'''
