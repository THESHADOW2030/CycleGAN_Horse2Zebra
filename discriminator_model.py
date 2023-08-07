import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, inChannels, outChannels, stride):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, 
                      out_channels=outChannels, 
                      kernel_size=4, 
                      stride=stride, 
                      padding=1, 
                      padding_mode="reflect", 
                      bias=True),
            nn.InstanceNorm2d(outChannels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)
    

class Discriminator(nn.Module):
    def __init__(self, inChannels = 3, features = [64, 128, 256, 512]):
        super(Discriminator, self).__init__()



        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True)
        )
        

        layers = []
        inChannels = features[0]
        for feature in features[1:]:
            layers.append(Block(inChannels, feature, stride=1 if feature == features[-1] else 2))
            inChannels = feature
        layers.append(nn.Conv2d(in_channels=inChannels, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        
        
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        return torch.sigmoid(x)


def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(inChannels=3)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()