import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, down=True, use_act = True, **kwargs):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, **kwargs) if down else nn.ConvTranspose2d(inChannels, outChannels, **kwargs),
            nn.InstanceNorm2d(outChannels), 
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
    

    def forward(self, x):
        return self.conv(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, channcels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(ConvBlock(channcels, channcels, kernel_size = 3, padding = 1),
                                   ConvBlock(channcels, channcels, use_act = False, kernel_size = 3, padding = 1)
                                    )
        
    def forward(self, x):
        return x + self.block(x)
    

class Generator(nn.Module):
    def __init__(self, imgChannels, numFeatures = 64, numResiduals = 9):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(imgChannels, numFeatures, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )

        self.downBlocks = nn.ModuleList([   #for downsampling
            ConvBlock(numFeatures, numFeatures*2, kernel_size = 3, stride = 2, padding = 1),
            ConvBlock(numFeatures*2, numFeatures*4,kernel_size = 3, stride = 2, padding = 1 )
        ])

        self.residualBlocks = nn.Sequential(*[ResidualBlock(numFeatures*4) for _ in range(numResiduals)])

        self.upBlocks = nn.ModuleList([
            ConvBlock(numFeatures * 4, numFeatures  * 2, down = False, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            ConvBlock(numFeatures * 2, numFeatures * 1, down = False, kernel_size = 3, stride = 2, padding = 1, output_padding = 1) 

        ])

        self.last = nn.Conv2d(numFeatures * 1, imgChannels, kernel_size = 7, stride = 1, padding = 3, padding_mode = "reflect")


    def forward(self, x):
        x = self.initial(x)
        for layer in self.downBlocks:
            x = layer(x)
        x = self.residualBlocks(x)
        for layer in self.upBlocks:
            x = layer(x)
        return torch.tanh(self.last(x))
    



def test():
    imgChannels = 3
    imgSize = 256
    x = torch.randn((2, imgChannels, imgSize, imgSize))
    gen  = Generator(imgChannels, 9)
   
    print(gen(x).shape)


if __name__ == "__main__":
    test()