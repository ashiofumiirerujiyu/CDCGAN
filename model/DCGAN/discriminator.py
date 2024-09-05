import torch.nn as nn


class DCGANDiscriminator(nn.Module):
    def __init__(self, img_dim, feature=64, dropout_p=0.5):
        super(DCGANDiscriminator, self).__init__()
        self.dropout_p = dropout_p
        self.net = nn.Sequential(
            self.conv_block(img_dim, feature, 4, 2, 1, batch_norm=False),
            self.conv_block(feature, feature * 2, 4, 2, 1),
            self.conv_block(feature * 2, feature * 4, 4, 2, 1),
            self.conv_block(feature * 4, feature * 8, 4, 2, 1),
            nn.Conv2d(feature * 8, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout2d(self.dropout_p))

        return nn.Sequential(*layers)

    def forward(self, img):
        return self.net(img).view(-1)
    