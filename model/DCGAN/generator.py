import torch.nn as nn


class DCGANGenerator(nn.Module):
    def __init__(self, noise_dim, img_dim, feature=64, dropout_p=0.5):
        super(DCGANGenerator, self).__init__()
        self.dropout_p = dropout_p
        self.net = nn.Sequential(
            self.deconv_block(noise_dim, feature * 8, 4, 1, 0),
            self.deconv_block(feature * 8, feature * 4, 4, 2, 1),
            self.deconv_block(feature * 4, feature * 2, 4, 2, 1),
            self.deconv_block(feature * 2, feature, 4, 2, 1),
            nn.ConvTranspose2d(feature, img_dim, 4, 2, 1),
            nn.Tanh()
        )

    def deconv_block(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0, batch_norm=True):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout2d(self.dropout_p))

        return nn.Sequential(*layers)
    
    def forward(self, noise):
        return self.net(noise)
    